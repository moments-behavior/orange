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
#include "global.h"
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
    case camnet::v1::ServerControl_STARTSTREAMING:
        return "STARTSTREAMING";
    case camnet::v1::ServerControl_BUMBLEBEEBOARD:
        return "BUMBLEBEEBOARD";
    case camnet::v1::ServerControl_TAKEPICTURE:
        return "TAKEPICTURE";
    case camnet::v1::ServerControl_NEXTPOSE:
        return "NEXTPOSE";
    case camnet::v1::ServerControl_GRIMLOCKBOARD:
        return "GRIMLOCKBOARD";
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
static uint32_t g_last_done = 0;
static std::string g_job;
static uint32_t g_epoch = 0;

static bool accept_session(const camnet::v1::Server *cmd) {
    if (!cmd)
        return false;
    const std::string job = cmd->job_id() ? cmd->job_id()->str() : "";
    if (g_job.empty() || cmd->epoch() > g_epoch || job != g_job) {
        g_job = job;
        g_epoch = cmd->epoch();
        g_last_done = 0;
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
static PTPParams ptp_params{0, 0, 0, 0, false, false, false, false};

static void cleanup_host_server_resources() {
    ptp_params.network_set_stop_ptp = false;
    for (auto &t : camera_threads)
        if (t.joinable())
            t.join();
    camera_threads.clear();

    for (int i = 0; i < cam_count; i++) {
        ptp_sync_off(&ecams[i].camera, &cameras_params[i]);
    }
    ptp_params.ptp_global_time = 0;
    ptp_params.ptp_stop_time = 0;
    ptp_params.ptp_counter = 0;
    ptp_params.ptp_stop_counter = 0;
    ptp_params.network_sync = false;
    ptp_params.network_set_start_ptp = false;
    ptp_params.ptp_stop_reached = false;
    ptp_params.ptp_start_reached = false;
    camera_control.sync_camera = false;

    for (int i = 0; i < cam_count; i++) {
        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                             evt_buffer_size, &cameras_params[i]);
        delete[] ecams[i].evt_frame;
        check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera),
                            cameras_params[i].camera_serial.c_str());
        close_camera(&ecams[i].camera, &cameras_params[i]);
    }
    ecams.clear();
    cameras_params.clear();
    cameras_select.clear();

    g_last_done = 0;
}

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
    ptp_params.network_sync = true;

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
            camera_threads.push_back(std::thread(
                &acquire_frames, &ecams[i], &cameras_params[i],
                &cameras_select[i], &camera_control, nullptr,
                encoder_basic_setup, record_folder, &ptp_params, nullptr));
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

static bool start_camera_streaming(std::string calib_folder,
                                   std::string save_format) {

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

    camera_control.record_video = false;
    camera_control.subscribe = true;
    camera_control.sync_camera = true;
    ptp_params.network_sync = true;

    if (!make_folder(calib_folder)) {
        std::cout << "Error creating calib_folder." << std::endl;
        return false;
    }

    for (int i = 0; i < cam_count; i++) {
        ptp_camera_sync(&ecams[i].camera, &cameras_params[i]);
    }

    for (int i = 0; i < cam_count; i++) {
        cameras_select[i].stream_on = false;
        cameras_select[i].picture_save_folder = calib_folder;
        cameras_select[i].frame_save_format = save_format;
    }

    try {
        for (int i = 0; i < cam_count; i++) {
            camera_threads.push_back(
                std::thread(&acquire_frames, &ecams[i], &cameras_params[i],
                            &cameras_select[i], &camera_control, nullptr, "",
                            "", &ptp_params, nullptr));
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

static bool take_picture(uint picture_id) {
    for (int i = 0; i < cam_count; i++) {
        cameras_select[i].pictures_counter = picture_id;
        cameras_select[i].frame_save_name =
            std::to_string(cameras_select[i].pictures_counter);
        cameras_select[i].sigs->frame_save_state.store(State_Copy_New_Frame);
    }

    using namespace std::chrono;
    auto t0 = steady_clock::now();
    const auto timeout = 180s; // tune for your hardware

    auto all_ready = [&] {
        for (int i = 0; i < cam_count; ++i) {
            if (cameras_select[i].sigs->frame_save_state.load(
                    std::memory_order_acquire) != State_Frame_Idle) {
                return false;
            }
        }
        return true;
    };

    while (!all_ready()) {
        if (steady_clock::now() - t0 > timeout) {
            // give up gracefully: flip subscribe so threads can finish politely
            camera_control.subscribe = false;
            return false;
        }
        // small sleep to avoid busy-spinning the core
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

static bool ctrl_action(camnet::v1::ServerControl c,
                        const camnet::v1::Server *msg) {
    switch (c) {
    case camnet::v1::ServerControl_OPENCAMERA: {
        auto *oa = msg->command_body_as_OpenArgs();
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
        auto *sta = msg->command_body_as_StartThreadsArgs();
        if (!sta)
            return false;
        std::string record_folder = sta->record_folder()->str();
        std::string encoder_setup = sta->encoder_setup()->str();
        std::cout << record_folder << std::endl;
        std::cout << encoder_setup << std::endl;
        return start_camera_thread(record_folder, encoder_setup);
    }

    case camnet::v1::ServerControl_STARTRECORDING: {
        auto *record_start = msg->command_body_as_StartArgs();
        if (!record_start)
            return false;
        unsigned long long ptp_global_time = record_start->ptp_time();
        std::cout << ptp_global_time << std::endl;
        ptp_params.ptp_global_time = ptp_global_time;
        ptp_params.network_set_start_ptp = true;
        return true;
    }

    case camnet::v1::ServerControl_STOPRECORDING: {
        auto *record_stop = msg->command_body_as_StopArgs();
        if (!record_stop)
            return false;
        unsigned long long ptp_stop_time = record_stop->ptp_time();
        std::cout << ptp_stop_time << std::endl;
        ptp_params.ptp_stop_time = ptp_stop_time;
        ptp_params.network_set_stop_ptp = true;
        // check if it has stopped

        using namespace std::chrono;
        auto t0 = steady_clock::now();
        const auto timeout = 180s; // tune for your hardware

        while (!ptp_params.ptp_stop_reached) {
            if (steady_clock::now() - t0 > timeout) {
                cleanup_host_server_resources();
                return false; // guard joins threads
            }
            // small sleep to avoid busy-spinning the core
            std::this_thread::sleep_for(1ms);
        }
        cleanup_host_server_resources();
        return true;
    }

    case camnet::v1::ServerControl_STARTSTREAMING: {
        auto *ssa = msg->command_body_as_StartStreamingArgs();
        if (!ssa)
            return false;
        std::string calib_folder = ssa->calib_folder()->str();
        std::string save_format = ssa->save_format()->str();
        std::cout << calib_folder << ", " << save_format << std::endl;
        return start_camera_streaming(calib_folder, save_format);
    }

    case camnet::v1::ServerControl_BUMBLEBEEBOARD: {
        return true;
    }

    case camnet::v1::ServerControl_TAKEPICTURE: {
        auto *tpa = msg->command_body_as_TakePictureArgs();
        if (!tpa)
            return false;
        uint picture_id = tpa->picture_id();
        return take_picture(picture_id);
    }

    case camnet::v1::ServerControl_NEXTPOSE: {
        return true;
    }

    case camnet::v1::ServerControl_GRIMLOCKBOARD: {
        return true;
    }

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
        if (cmd->seq() <= g_last_done) {
            auto bytes = build_phase_reply(cmd, g_name, 2, true, 0, "already");
            send_bytes(evt.peer_id, bytes);
            std::printf("[SRV %s] %s (duplicate)\n", g_name.c_str(),
                        ctrl_name(ctrl));
            break;
        }

        bool ok = ctrl_action(ctrl, cmd);
        // Reply reflects actual outcome
        auto bytes = build_phase_reply(cmd, g_name, cam_count, ok, ok ? 0 : -1,
                                       ok ? "ok" : "failed");
        send_bytes(evt.peer_id, bytes);
        // Only advance monotonic seq on success
        if (ok)
            g_last_done = cmd->seq();

        std::printf("[SRV %s] %s%s\n", g_name.c_str(), ctrl_name(ctrl),
                    ok ? "" : " (failed)");
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
