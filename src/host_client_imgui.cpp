#include "host_client_imgui.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ctrl_generated.h"       // camnet::v1 schema (generated)
#include "enet_fb_helpers.h"      // optional FB parse helpers
#include "enet_runtime_unified.h" // <-- unified wrapper (connect/dispatch/start)
#include "enet_utils.h"           // AppContext, Incoming/Outgoing, PeerRegistry
#include "host_ctx.h"             // HostOpenCtx
#include "imgui.h"
#include "utils.h"

using namespace std::chrono_literals;

// ============================================================================
// Host-side OPEN phase context binding
// ============================================================================
static HostOpenCtx *g_openctx = nullptr;
void HostClient_SetOpenCtx(HostOpenCtx *ctx) { g_openctx = ctx; }

// ============================================================================
// Phase hooks
// ============================================================================
static void OnOpenPhaseStart() {
    if (!g_openctx)
        return;

    auto &selected_folder = *g_openctx->selected_network_folder;
    auto *devs = g_openctx->device_info;
    int &cam_count = *g_openctx->cam_count;
    auto &check = *g_openctx->check;

    int &num_cams = *g_openctx->num_cameras;
    CameraParams *&params = *g_openctx->cameras_params;
    CameraEachSelect *&select = *g_openctx->cameras_select;
    CameraEmergent *&ecams = *g_openctx->ecams;
    ScrollingBuffer *&plots = *g_openctx->realtime_plot_data;
    CameraControl *ctrl = g_openctx->camera_control;

    std::vector<std::string> cfg_files;
    update_camera_configs(cfg_files, selected_folder);
    select_cameras_have_configs(cfg_files, devs, check, cam_count);
    open_selected_cameras(check, cam_count, devs, cfg_files, num_cams, params,
                          select, ecams, plots);
    if (ctrl)
        ctrl->open = true;
}

static void OnOpenPhaseComplete() {
    // Optional: persist metadata, update UI, etc.
}

// ============================================================================
// FlatBuffers parse helper
// ============================================================================
static bool parse_server(const std::vector<uint8_t> &bytes,
                         const camnet::v1::Server *&out) {
    out = nullptr;
    if (bytes.empty())
        return false;
    flatbuffers::Verifier v(bytes.data(), bytes.size());
    if (!camnet::v1::VerifyServerBuffer(v))
        return false;
    out = camnet::v1::GetServer(bytes.data());
    return out != nullptr;
}

// ============================================================================
// Command builders
// ============================================================================
static std::vector<uint8_t> build_cmd_open(const std::string &job_id,
                                           uint32_t epoch, uint32_t seq,
                                           const std::string &config_folder) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto jid = b.CreateString(job_id);
    auto cfg = b.CreateString(config_folder);
    auto args = CreateOpenArgs(b, cfg);
    auto msg = CreateServer(b, Kind_KindCommand, ServerControl_OPENCAMERA, jid,
                            epoch, seq, CommandBody_OpenArgs, args.Union(), 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t>
build_cmd_startthreads(const std::string &job_id, uint32_t epoch, uint32_t seq,
                       const std::string &record_folder,
                       const std::string &encoder_setup) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto jid = b.CreateString(job_id);
    auto rec = b.CreateString(record_folder);
    auto enc = b.CreateString(encoder_setup);
    auto args = CreateStartThreadsArgs(b, rec, enc);
    auto msg =
        CreateServer(b, Kind_KindCommand, ServerControl_STARTTHREAD, jid, epoch,
                     seq, CommandBody_StartThreadsArgs, args.Union(), 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t> build_cmd_start(const std::string &job_id,
                                            uint32_t epoch, uint32_t seq,
                                            uint64_t ptp) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(128);
    auto jid = b.CreateString(job_id);
    auto args = CreateStartArgs(b, ptp);
    auto msg =
        CreateServer(b, Kind_KindCommand, ServerControl_STARTRECORDING, jid,
                     epoch, seq, CommandBody_StartArgs, args.Union(), 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t> build_cmd_stop(const std::string &job_id,
                                           uint32_t epoch, uint32_t seq,
                                           uint64_t ptp) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(128);
    auto jid = b.CreateString(job_id);
    auto args = CreateStopArgs(b, ptp);
    auto msg =
        CreateServer(b, Kind_KindCommand, ServerControl_STOPRECORDING, jid,
                     epoch, seq, CommandBody_StopArgs, args.Union(), 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

// ============================================================================
// Names / log helpers
// ============================================================================
static const char *ctrl_name(camnet::v1::ServerControl c) {
    switch (c) {
    case camnet::v1::ServerControl_OPENCAMERA:
        return "OPENCAMERA";
    case camnet::v1::ServerControl_STARTTHREAD:
        return "STARTTHREAD";
    case camnet::v1::ServerControl_STARTRECORDING:
        return "STARTRECORDING";
    case camnet::v1::ServerControl_STOPRECORDING:
        return "STOPRECORDING";
    default:
        return "NONE";
    }
}

// ============================================================================
// Global host client state
// ============================================================================
static AppContext *g_ctxp = nullptr;

enum Phase { Phase_Open, Phase_Threads, Phase_Start, Phase_Stop, Phase_Done };

static std::vector<std::pair<std::string, int>> g_endpoints; // host:port pairs
static std::vector<std::string> g_servers; // server names via bringup
static std::unordered_map<std::string, bool> g_ack_by;

static std::string g_jid = "job-demo-001";
static uint32_t g_epoch = 1;
static uint32_t g_seq = 1;
static Phase g_phase = Phase_Open;

static bool g_waiting = false;
static int g_timeout_ms = 2000;
static std::chrono::steady_clock::time_point g_last_send;

// We default to NOT stepping ENet in the GUI tick; a dispatcher thread handles
// it.
static bool g_step_net_in_tick = false;

// ---- thread-safe reply queue (producer: net thread, consumer: GUI)
struct ReplyEvent {
    camnet::v1::ServerControl ctrl{};
    std::string server_id;
    bool ok{};
    std::string job_id;
    uint32_t epoch{};
    uint32_t seq{};
    std::string detail;
};
static std::mutex g_rep_m;
static std::vector<ReplyEvent> g_rep_q;
static void push_reply(const ReplyEvent &e) {
    std::lock_guard<std::mutex> lk(g_rep_m);
    g_rep_q.push_back(e);
}
static void drain_replies(std::vector<ReplyEvent> &out) {
    std::lock_guard<std::mutex> lk(g_rep_m);
    out.swap(g_rep_q);
}

// ---- thread-safe logs (net thread may log)
static std::mutex g_logs_m;
static std::vector<std::string> g_logs;
static void logf(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> lk(g_logs_m);
    g_logs.emplace_back(buf);
}

static bool g_open_started = false;

// ============================================================================
// Utilities
// ============================================================================
static void update_servers_from_registry() {
    auto info = g_ctxp->peers.snapshot_info();
    std::vector<std::string> names;
    names.reserve(info.size());
    for (auto &pi : info)
        if (!pi.name.empty())
            names.push_back(pi.name);
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    g_servers = names;
}

static camnet::v1::ServerControl current_ctrl() {
    switch (g_phase) {
    case Phase_Open:
        return camnet::v1::ServerControl_OPENCAMERA;
    case Phase_Threads:
        return camnet::v1::ServerControl_STARTTHREAD;
    case Phase_Start:
        return camnet::v1::ServerControl_STARTRECORDING;
    case Phase_Stop:
        return camnet::v1::ServerControl_STOPRECORDING;
    default:
        return camnet::v1::ServerControl_NONE;
    }
}

static const char *phase_name() {
    switch (g_phase) {
    case Phase_Open:
        return "OPENCAMERA";
    case Phase_Threads:
        return "STARTTHREAD";
    case Phase_Start:
        return "STARTRECORDING";
    case Phase_Stop:
        return "STOPRECORDING";
    case Phase_Done:
        return "DONE";
    default:
        return "?";
    }
}

static void advance_phase() {
    switch (g_phase) {
    case Phase_Open:
        g_phase = Phase_Threads;
        break;
    case Phase_Threads:
        g_phase = Phase_Start;
        break;
    case Phase_Start:
        g_phase = Phase_Stop;
        break;
    case Phase_Stop:
        g_phase = Phase_Done;
        break;
    default:
        break;
    }
}

static void reset_session() {
    g_jid = "job-demo-001";
    g_epoch = 1;
    g_seq = 1;
    g_phase = Phase_Open;
    g_waiting = false;
    g_ack_by.clear();
    {
        std::lock_guard<std::mutex> lk(g_logs_m);
        g_logs.clear();
    }
    g_open_started = false;
    logf("session reset");
}

// ============================================================================
// Send helpers
// ============================================================================
static void send_bytes(uint32_t pid, const std::vector<uint8_t> &buf) {
    Outgoing o;
    o.peer_id = pid;
    o.channel = 0;
    o.flags = ENET_PACKET_FLAG_RELIABLE;
    o.bytes = buf;
    g_ctxp->net.send(o);
}

static void broadcast_current_phase() {
    if (g_phase == Phase_Done || g_servers.empty())
        return;

    std::vector<uint8_t> bytes;
    switch (g_phase) {
    case Phase_Open: {
        bytes = build_cmd_open(g_jid, g_epoch, g_seq,
                               *g_openctx->selected_network_folder);
        if (!g_open_started) {
            OnOpenPhaseStart();
            g_open_started = true;
        }
        break;
    }
    case Phase_Threads:
        bytes = build_cmd_startthreads(g_jid, g_epoch, g_seq, "out/session-1",
                                       "h264");
        break;
    case Phase_Start:
        bytes = build_cmd_start(g_jid, g_epoch, g_seq, 123456789ULL);
        break;
    case Phase_Stop:
        bytes = build_cmd_stop(g_jid, g_epoch, g_seq, 123457789ULL);
        break;
    default:
        break;
    }

    logf("-> %s (job=%s epoch=%u seq=%u)", ctrl_name(current_ctrl()),
         g_jid.c_str(), g_epoch, g_seq);
    for (const auto &name : g_servers) {
        uint32_t pid = g_ctxp->peers.get_pid_by_name(name);
        if (pid)
            send_bytes(pid, bytes);
    }
}

// ============================================================================
// ENet event handler (dispatcher / GUI fallback)
// ============================================================================
static void host_on_event(const Incoming &evt) {
    switch (evt.type) {
    case Incoming::Connect:
        g_ctxp->peers.add(evt.peer_id);
        logf("connected pid=%u", evt.peer_id);
        break;

    case Incoming::Disconnect:
        g_ctxp->peers.remove(evt.peer_id);
        logf("disconnected pid=%u", evt.peer_id);
        break;

    case Incoming::Receive: {
        const camnet::v1::Server *msg = nullptr;
        if (!parse_server(evt.bytes, msg))
            break;
        if (msg->kind() != camnet::v1::Kind_KindReply)
            break;
        auto rep = msg->reply();
        if (!rep)
            break;

        // bringup
        if (auto br = rep->bringup()) {
            const std::string name =
                br->server_name() ? br->server_name()->str() : "";
            g_ctxp->peers.set_bringup(evt.peer_id, name, br->num_cameras());
            logf("bringup from %s cams=%d (pid=%u)", name.c_str(),
                 br->num_cameras(), evt.peer_id);
        }

        // phase reply -> queue
        ReplyEvent e;
        e.ctrl = msg->control();
        e.server_id = rep->server_id() ? rep->server_id()->str() : "";
        e.ok = rep->ok();
        e.job_id = msg->job_id() ? msg->job_id()->str() : "";
        e.epoch = msg->epoch();
        e.seq = msg->seq();
        e.detail = rep->detail() ? rep->detail()->str() : "";
        push_reply(e);
        break;
    }

    default:
        break;
    }
}

// ============================================================================
// Net dispatcher thread
// ============================================================================
static std::atomic<bool> g_net_run{false};
static std::thread g_net_thr;

static void HostClient_NetThreadMain(AppContext *ctxptr) {
    if (!ctxptr)
        return;
    while (g_net_run.load(std::memory_order_relaxed)) {
        int n = enet_dispatch_drain(ctxptr->net, host_on_event);
        if (n == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void HostClient_StartNetThread(AppContext &ctx) {
    if (g_net_run.exchange(true))
        return;
    g_net_thr = std::thread(HostClient_NetThreadMain, &ctx);
}

void HostClient_StopNetThread() {
    if (!g_net_run.exchange(false))
        return;
    if (g_net_thr.joinable())
        g_net_thr.join();
}

// ============================================================================
// Public API
// ============================================================================
void HostClient_SetStepInTick(bool v) { g_step_net_in_tick = v; }

void HostClient_Init(
    AppContext &ctx,
    const std::vector<std::pair<std::string, int>> &endpoints) {
    g_ctxp = &ctx;
    g_endpoints = endpoints;

    // ensure client host/I-O thread in threaded mode; no-op for inline
    if (!enet_start_client(g_ctxp->net, 2, 0, 0)) {
        std::fprintf(stderr, "[HOST] failed to start client runtime\n");
        return;
    }

    for (const auto &ep : g_endpoints) {
        ConnectReq cr{ep.first.c_str(), static_cast<uint16_t>(ep.second), 2, 0};
        if (!enet_connect(g_ctxp->net, cr)) {
            std::fprintf(stderr, "[HOST] connect enqueue failed for %s:%d\n",
                         ep.first.c_str(), ep.second);
        } else {
            logf("[HOST] dialing %s:%d", ep.first.c_str(), ep.second);
        }
    }

    g_last_send = std::chrono::steady_clock::now();
}

void HostClient_Tick() {
    if (g_step_net_in_tick) {
        (void)enet_dispatch_drain(g_ctxp->net, host_on_event);
    }

    update_servers_from_registry();

    std::vector<ReplyEvent> batch;
    drain_replies(batch);
    if (!batch.empty()) {
        for (const ReplyEvent &e : batch) {
            if (e.job_id != g_jid || e.epoch != g_epoch || e.seq != g_seq)
                continue;
            if (e.ctrl != current_ctrl())
                continue;
            logf("%s reply from %s ok=%d", ctrl_name(e.ctrl),
                 e.server_id.c_str(), (int)e.ok);
            if (e.ok)
                g_ack_by[e.server_id] = true;
        }

        if (g_waiting) {
            bool all = true;
            for (const auto &name : g_servers) {
                if (!g_ack_by.count(name) || !g_ack_by[name]) {
                    all = false;
                    break;
                }
            }
            if (all) {
                logf("barrier OK for %s", ctrl_name(current_ctrl()));
                if (current_ctrl() == camnet::v1::ServerControl_OPENCAMERA) {
                    OnOpenPhaseComplete();
                    g_open_started = false;
                }
                g_waiting = false;
                ++g_seq;
                advance_phase();
            }
        }
    }

    if (g_waiting) {
        auto now = std::chrono::steady_clock::now();
        if (now - g_last_send > std::chrono::milliseconds(g_timeout_ms)) {
            for (const auto &name : g_servers) {
                if (!g_ack_by.count(name) || !g_ack_by[name]) {
                    logf("waiting for %s on %s …", name.c_str(),
                         ctrl_name(current_ctrl()));
                }
            }
            broadcast_current_phase();
            g_last_send = std::chrono::steady_clock::now();
            logf("timeout → re-send %s", ctrl_name(current_ctrl()));
        }
    }
}

void HostClient_DrawImGui() {
    ImGui::Begin("Host Client");

    if (ImGui::CollapsingHeader("Endpoints", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto &ep : g_endpoints) {
            ImGui::BulletText("%s:%d", ep.first.c_str(), ep.second);
        }
    }

    if (ImGui::CollapsingHeader("Camera Servers",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        auto info = g_ctxp->peers.snapshot_info();
        ImGui::Text("Known peers: %d", (int)info.size());
        for (const auto &pi : info) {
            ImGui::BulletText("pid=%u name=%s camsK=%d cams=%d", pi.peer_id,
                              pi.name.c_str(), (int)pi.camsK, pi.cams);
        }
    }

    ImGui::Separator();
    ImGui::Text("Job: %s  epoch=%u  seq=%u", g_jid.c_str(), g_epoch, g_seq);
    ImGui::Text("Phase: %s", phase_name());

    if (!g_servers.empty()) {
        ImGui::Text("Acks:");
        ImGui::Indent();
        for (const auto &name : g_servers) {
            bool got = g_ack_by.count(name) ? g_ack_by[name] : false;
            ImGui::BulletText("%s  [%s]", name.c_str(), got ? "OK" : "...");
        }
        ImGui::Unindent();
    }

    if (ImGui::Button(g_waiting ? "Waiting..." : "Advance Phase")) {
        if (!g_waiting && g_phase != Phase_Done) {
            g_ack_by.clear();
            broadcast_current_phase();
            g_last_send = std::chrono::steady_clock::now();
            g_waiting = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Resend")) {
        broadcast_current_phase();
        g_last_send = std::chrono::steady_clock::now();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Session")) {
        reset_session();
    }

    ImGui::Separator();
    if (ImGui::Button("Clear Log")) {
        std::lock_guard<std::mutex> lk(g_logs_m);
        g_logs.clear();
    }
    ImGui::BeginChild("log", ImVec2(0, 0), true,
                      ImGuiWindowFlags_HorizontalScrollbar);
    {
        std::lock_guard<std::mutex> lk(g_logs_m);
        for (const auto &line : g_logs)
            ImGui::TextUnformatted(line.c_str());
    }
    ImGui::EndChild();

    ImGui::End();
}
