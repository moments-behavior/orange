#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "camera.h"
#include "ctrl_generated.h"
#include "enet_fb_helpers.h"
#include "enet_runtime_unified.h" // unified wrapper
#include "enet_utils.h"
#include "utils.h"
#include "video_capture.h"
#include <iostream>
#include <variant>
using namespace std::chrono_literals;

static AppContext *g_ctx = nullptr;
static std::string g_name;

// ---------- FlatBuffers parse ----------
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

// ---------- small helpers ----------
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

static void send_bytes(uint32_t pid, const std::vector<uint8_t> &buf) {
    Outgoing o;
    o.peer_id = pid;
    o.channel = 0;
    o.flags = ENET_PACKET_FLAG_RELIABLE;
    o.bytes = buf;
    g_ctx->net.send(o);
}

// ---------- replies ----------
static std::vector<uint8_t> build_bringup_reply(const std::string &name,
                                                uint16_t cam_count) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto sid = b.CreateString(name);
    auto br = CreateBringupMessage(b, sid, cam_count);

    auto rep = CreateReplyInfo(b,
                               true,                    // ok
                               0,                       // code
                               b.CreateString("hello"), // detail
                               br,                      // bringup
                               sid                      // server_id
    );

    auto srv = CreateServer(b, Kind_KindReply, ServerControl_NONE,
                            /*job_id*/ 0, /*epoch*/ 0, /*seq*/ 0,
                            CommandBody_NONE, 0, rep);
    b.Finish(srv);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t> build_phase_reply(const camnet::v1::Server *cmd,
                                              const std::string &name,
                                              uint16_t cam_count,
                                              bool ok = true, int code = 0,
                                              const char *detail = "ok") {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto sid = b.CreateString(name);
    auto det = b.CreateString(detail ? detail : "");

    auto rep = CreateReplyInfo(b, ok, code, det,
                               /*bringup*/ 0, sid);

    // Preserve job/epoch/seq from command (if present)
    flatbuffers::Offset<flatbuffers::String> jid =
        cmd->job_id() ? b.CreateString(cmd->job_id()->str())
                      : flatbuffers::Offset<flatbuffers::String>{};

    auto srv = CreateServer(b, Kind_KindReply, cmd->control(), jid,
                            cmd->epoch(), cmd->seq(), CommandBody_NONE, 0, rep);
    b.Finish(srv);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

// ---------- server session gate ----------
static camnet::v1::ServerControl g_last_done = camnet::v1::ServerControl_NONE;
static std::string g_job;
static uint32_t g_epoch = 0;

static bool accept_session(const camnet::v1::Server *cmd) {
    if (!cmd)
        return false;
    const std::string job = cmd->job_id() ? cmd->job_id()->str() : "";
    if (g_job.empty() || cmd->epoch() > g_epoch || job != g_job) {
        g_job = job;
        g_epoch = cmd->epoch();
        g_last_done = camnet::v1::ServerControl_NONE;
    }
    if (cmd->epoch() < g_epoch)
        return false; // stale epoch
    return true;
}

static std::vector<GigEVisionDeviceInfo> unsorted_devices;
static std::vector<GigEVisionDeviceInfo> sorted_devices;
static int max_cameras = 10;
static int evt_buffer_size = 100;
static int cam_count;
static std::vector<CameraEmergent> ecams;
static std::vector<CameraParams> cameras_params;
static std::vector<CameraEachSelect> cameras_select;
static CameraControl camera_control;
static std::vector<std::thread> camera_threads;
static PTPParams ptp_params{0, 0, 0, 0, true, false, false, false};

static bool open_cameras(const std::string &config_folder) {
    const size_t n = cameras_params.size();
    if (ecams.size() != n || cameras_select.size() != n)
        return false;
    if (sorted_devices.size() < n)
        return false;

    std::vector<std::string> camera_config_files;
    update_camera_configs(camera_config_files, config_folder);

    for (size_t i = 0; i < n; ++i) {
        set_camera_params(&cameras_params[i], &cameras_select[i],
                          &sorted_devices[i], camera_config_files,
                          static_cast<int>(i), static_cast<int>(n));

        open_camera_with_params(&ecams[i].camera,
                                &sorted_devices[cameras_params[i].camera_id],
                                &cameras_params[i]);
    }
    return true;
}

static bool start_camera_thread(std::string record_folder,
                                std::string encoder_basic_setup) {

    // RAII guard: if we exit false, join any threads we created.
    struct Joiner {
        std::vector<std::thread> &ts;
        bool disarm{false};
        ~Joiner() {
            if (!disarm)
                for (auto &t : ts)
                    if (t.joinable())
                        t.join();
        }
    } guard{camera_threads};

    try {
        // allocate frames
        for (int i = 0; i < cam_count; i++) {
            camera_open_stream(&ecams[i].camera, &cameras_params[i]);
            ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
            allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                                  &cameras_params[i], evt_buffer_size);
            if (cameras_params[i].need_reorder &&
                cameras_params[i].gpu_direct) {
                allocate_frame_reorder_buffer(&ecams[i].camera,
                                              &ecams[i].frame_reorder,
                                              &cameras_params[i]);
            }
        }

    } catch (...) {
        std::cout << "Error allocating camera frame buffers." << std::endl;
        return false;
    }

    camera_control.record_video = true;
    camera_control.subscribe = true;
    camera_control.sync_camera = true;

    if (!make_folder(record_folder)) {
        std::cout << "Error creating recording folder." << std::endl;
        return false;
    }

    for (int i = 0; i < cam_count; i++) {
        ptp_camera_sync(&ecams[i].camera, &cameras_params[i]);
    }

    for (int i = 0; i < cam_count; i++) {
        cameras_select[i].stream_on = false;
    }
    try {
        for (int i = 0; i < cam_count; i++) {
            camera_threads.push_back(
                std::thread(&acquire_frames, &ecams[i], &cameras_params[i],
                            &cameras_select[i], &camera_control, nullptr,
                            encoder_basic_setup, record_folder, &ptp_params));
        }
    } catch (...) {
        std::cout << "Error creating camera thread." << std::endl;
        return false;
    }

    using namespace std::chrono;
    auto t0 = steady_clock::now();
    const auto timeout = 180s; // tune for your hardware

    while (ptp_params.ptp_counter != cam_count) {
        if (steady_clock::now() - t0 > timeout) {
            // give up gracefully: flip subscribe so threads can finish politely
            camera_control.subscribe = false;
            return false; // guard joins threads
        }
        // small sleep to avoid busy-spinning the core
        std::this_thread::sleep_for(1ms);
    }

    // success
    guard.disarm = true;
    return true;
}

static bool ctrl_action(camnet::v1::ServerControl c,
                        const camnet::v1::Server *msg) {
    switch (c) {
    case camnet::v1::ServerControl_OPENCAMERA: {
        const camnet::v1::OpenArgs *oa = msg->command_body_as_OpenArgs();
        if (!oa)
            return false; // union wasn't OpenArgs

        const auto *s =
            oa->config_folder(); // flatbuffers::String* (may be null)
        if (!s)
            return false;

        std::string config_folder = s->str(); // std::string copy

        cameras_params.resize(cam_count);
        ecams.resize(cam_count);
        cameras_select.resize(cam_count);

        return open_cameras(config_folder);
    }

    case camnet::v1::ServerControl_STARTTHREAD: {
        const camnet::v1::StartThreadsArgs *sta =
            msg->command_body_as_StartThreadsArgs();
        if (!sta)
            return false;
        std::string record_folder = sta->record_folder()->str();
        std::string encoder_setup = sta->encoder_setup()->str();
        std::cout << record_folder << std::endl;
        std::cout << encoder_setup << std::endl;
        return start_camera_thread(record_folder, encoder_setup);
    }

    case camnet::v1::ServerControl_STARTRECORDING: {
        return true;
        // start_recording(a->filename); // implement this
    }

    case camnet::v1::ServerControl_STOPRECORDING:
        return true;
        // stop_recording(); // implement this

    default:
        return false;
    }
}

// ---------- event handler ----------
static void server_on_event(const Incoming &evt) {
    switch (evt.type) {
    case Incoming::Connect: {
        std::printf("[SRV %s] host connected (pid=%u) → sending bringup\n",
                    g_name.c_str(), evt.peer_id);

        unsorted_devices.clear();
        sorted_devices.clear();
        unsorted_devices.resize(max_cameras);
        sorted_devices.resize(max_cameras);

        cam_count = scan_cameras(max_cameras, unsorted_devices.data());
        sort_cameras_ip(unsorted_devices.data(), sorted_devices.data(),
                        cam_count);
        auto bytes = build_bringup_reply(g_name, /*cams*/ cam_count);
        send_bytes(evt.peer_id, bytes);
        break;
    }
    case Incoming::Receive: {
        const camnet::v1::Server *cmd = nullptr;
        if (!parse_server(evt.bytes, cmd))
            break;
        if (!cmd || cmd->kind() != camnet::v1::Kind_KindCommand)
            break;
        if (!accept_session(cmd))
            break;

        auto ctrl = cmd->control();

        // Idempotent: if we already completed this or a later phase, ack
        // again
        if (ctrl <= g_last_done) {
            auto bytes = build_phase_reply(cmd, g_name, 2, true, 0, "already");
            send_bytes(evt.peer_id, bytes);
            std::printf("[SRV %s] %s (duplicate)\n", g_name.c_str(),
                        ctrl_name(ctrl));
            break;
        }

        bool is_success = ctrl_action(ctrl, cmd);
        // do we send reply if it is failure?
        auto bytes = build_phase_reply(cmd, g_name, cam_count, true, 0, "ok");
        send_bytes(evt.peer_id, bytes);
        g_last_done = ctrl;
        std::printf("[SRV %s] %s\n", g_name.c_str(), ctrl_name(ctrl));
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
        // Wait up to 5 ms (inline) or sleep/poll (threaded), then deliver
        // events
        enet_dispatch_block(ctx.net, 5, server_on_event);
        // ... do per-tick housekeeping here if needed ...
    }
    return 0;
}
