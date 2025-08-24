#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "ctrl_generated.h"
#include "enet_fb_helpers.h"
#include "enet_runtime_unified.h"  // unified wrapper
#include "enet_utils.h"

using namespace std::chrono_literals;

static AppContext *g_ctx = nullptr;
static std::string g_name;

// ---------- FlatBuffers parse ----------
static bool parse_server(const std::vector<uint8_t> &bytes,
                         const camnet::v1::Server *&out) {
    out = nullptr;
    if (bytes.empty()) return false;
    flatbuffers::Verifier v(bytes.data(), bytes.size());
    if (!camnet::v1::VerifyServerBuffer(v)) return false;
    out = camnet::v1::GetServer(bytes.data());
    return out != nullptr;
}

// ---------- small helpers ----------
static const char *ctrl_name(camnet::v1::ServerControl c) {
    switch (c) {
    case camnet::v1::ServerControl_OPENCAMERA:     return "OPENCAMERA";
    case camnet::v1::ServerControl_STARTTHREAD:    return "STARTTHREAD";
    case camnet::v1::ServerControl_STARTRECORDING: return "STARTRECORDING";
    case camnet::v1::ServerControl_STOPRECORDING:  return "STOPRECORDING";
    default:                                       return "NONE";
    }
}

static const char *state_name(camnet::v1::ManagerState s) {
    switch (s) {
    case camnet::v1::ManagerState_IDLE:              return "IDLE";
    case camnet::v1::ManagerState_CONNECTED:         return "CONNECTED";
    case camnet::v1::ManagerState_CAMERAOPENED:      return "CAMERAOPENED";
    case camnet::v1::ManagerState_THREADREADY:       return "THREADREADY";
    case camnet::v1::ManagerState_RECORDINGSTARTED:  return "RECORDINGSTARTED";
    case camnet::v1::ManagerState_RECORDSTOPPED:     return "RECORDSTOPPED";
    case camnet::v1::ManagerState_ERROR:             return "ERROR";
    default:                                         return "?";
    }
}

static camnet::v1::ManagerState state_after(camnet::v1::ServerControl c) {
    switch (c) {
    case camnet::v1::ServerControl_OPENCAMERA:     return camnet::v1::ManagerState_CAMERAOPENED;
    case camnet::v1::ServerControl_STARTTHREAD:    return camnet::v1::ManagerState_THREADREADY;
    case camnet::v1::ServerControl_STARTRECORDING: return camnet::v1::ManagerState_RECORDINGSTARTED;
    case camnet::v1::ServerControl_STOPRECORDING:  return camnet::v1::ManagerState_RECORDSTOPPED;
    default:                                       return camnet::v1::ManagerState_ERROR;
    }
}

static void send_bytes(uint32_t pid, const std::vector<uint8_t> &buf) {
    Outgoing o;
    o.peer_id = pid;
    o.channel = 0;
    o.flags   = ENET_PACKET_FLAG_RELIABLE;
    o.bytes   = buf;
    g_ctx->net.send(o);
}

// ---------- replies ----------
static std::vector<uint8_t> build_bringup_reply(const std::string &name,
                                                uint16_t cam_count) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto sid = b.CreateString(name);
    auto br  = CreateBringupMessage(b, sid, cam_count);

    auto rep = CreateReplyInfo(
        b,
        ManagerState_CONNECTED,        // initial state
        true,                          // ok
        0,                             // code
        b.CreateString("hello"),       // detail
        br,                            // bringup
        cam_count,                     // num_cameras
        sid                            // server_id
    );

    auto srv = CreateServer(
        b, Kind_KindReply, ServerControl_NONE,
        /*job_id*/ 0, /*epoch*/ 0, /*seq*/ 0,
        CommandBody_NONE, 0,
        rep
    );
    b.Finish(srv);
    return { b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize() };
}

static std::vector<uint8_t>
build_phase_reply(const camnet::v1::Server *cmd, const std::string &name,
                  uint16_t cam_count, camnet::v1::ManagerState st,
                  bool ok = true, int code = 0, const char *detail = "ok") {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto sid = b.CreateString(name);
    auto det = b.CreateString(detail ? detail : "");

    auto rep = CreateReplyInfo(
        b, st, ok, code, det,
        /*bringup*/ 0,
        cam_count,
        sid
    );

    // Preserve job/epoch/seq from command (if present)
    flatbuffers::Offset<flatbuffers::String> jid =
        cmd->job_id() ? b.CreateString(cmd->job_id()->str())
                      : flatbuffers::Offset<flatbuffers::String>{};

    auto srv = CreateServer(
        b,
        Kind_KindReply,
        cmd->control(),
        jid,
        cmd->epoch(),
        cmd->seq(),
        CommandBody_NONE,
        0,
        rep
    );
    b.Finish(srv);
    return { b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize() };
}

// ---------- server session gate ----------
static camnet::v1::ServerControl g_last_done = camnet::v1::ServerControl_NONE;
static std::string g_job;
static uint32_t g_epoch = 0;

static bool accept_session(const camnet::v1::Server *cmd) {
    if (!cmd) return false;
    const std::string job = cmd->job_id() ? cmd->job_id()->str() : "";
    if (g_job.empty() || cmd->epoch() > g_epoch || job != g_job) {
        g_job = job;
        g_epoch = cmd->epoch();
        g_last_done = camnet::v1::ServerControl_NONE;
    }
    if (cmd->epoch() < g_epoch) return false; // stale epoch
    return true;
}

// ---------- event handler ----------
static void server_on_event(const Incoming &evt) {
    switch (evt.type) {
    case Incoming::Connect: {
        std::printf("[SRV %s] host connected (pid=%u) → sending bringup\n",
                    g_name.c_str(), evt.peer_id);
        auto bytes = build_bringup_reply(g_name, /*cams*/ 2);
        send_bytes(evt.peer_id, bytes);
        break;
    }
    case Incoming::Receive: {
        const camnet::v1::Server *cmd = nullptr;
        if (!parse_server(evt.bytes, cmd)) break;
        if (!cmd || cmd->kind() != camnet::v1::Kind_KindCommand) break;
        if (!accept_session(cmd)) break;

        auto ctrl = cmd->control();

        // Idempotent: if we already completed this or a later phase, ack again
        if (ctrl <= g_last_done) {
            auto st = state_after(ctrl);
            auto bytes = build_phase_reply(cmd, g_name, 2, st, true, 0, "already");
            send_bytes(evt.peer_id, bytes);
            std::printf("[SRV %s] %s (duplicate) -> %s\n",
                        g_name.c_str(), ctrl_name(ctrl), state_name(st));
            break;
        }

        // COMM ONLY: pretend success immediately; integrate real work here
        auto st = state_after(ctrl);
        auto bytes = build_phase_reply(cmd, g_name, 2, st, true, 0, "ok");
        send_bytes(evt.peer_id, bytes);
        g_last_done = ctrl;
        std::printf("[SRV %s] %s -> %s\n",
                    g_name.c_str(), ctrl_name(ctrl), state_name(st));
        break;
    }
    case Incoming::Disconnect:
        std::printf("[SRV %s] host disconnected\n", g_name.c_str());
        break;
    }
}

int main(int argc, char **argv) {
    AppContext ctx;
    g_ctx = &ctx;

    // args: cam_server <name> [port]
    g_name = (argc > 1) ? argv[1] : std::string("camA");
    int port = (argc > 2) ? std::atoi(argv[2]) : 34001;

    if (!ctx.net.start_server(static_cast<uint16_t>(port))) {
        std::fprintf(stderr, "[%s] Failed to start ENet server on :%d\n",
                     g_name.c_str(), port);
        return 1;
    }
    std::printf("[SRV %s] listening on :%d\n", g_name.c_str(), port);

    // Unified dispatch: works for both inline and threaded runtimes
    for (;;) {
        // Wait up to 5 ms (inline) or sleep/poll (threaded), then deliver events
        enet_dispatch_block(ctx.net, 5, server_on_event);
        // ... do per-tick housekeeping here if needed ...
    }
    return 0;
}
