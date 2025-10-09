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

#include "camera.h"
#include "ctrl_generated.h"       // camnet::v1 schema (generated)
#include "enet_fb_helpers.h"      // optional FB parse helpers
#include "enet_runtime_unified.h" // <-- unified wrapper (connect/dispatch/start)
#include "enet_utils.h"           // AppContext, Incoming/Outgoing, PeerRegistry
#include "gui.h"
#include "imgui.h"
#include "utils.h"
using namespace std::chrono_literals;

// persistent to handle missing messages
static uint g_picture_id;
static std::string g_folder_name;
static unsigned long long g_ptp_start_time;
static unsigned long long g_ptp_stop_time;

static HostClientCtx *g_clientctx = nullptr;
void set_host_client_ctx(HostClientCtx *ctx) { g_clientctx = ctx; }

// ============================================================================
// Phase hooks
// ============================================================================
static void on_open_phase_start() {
    std::string &selected_folder = *g_clientctx->selected_network_folder;
    auto *devs = g_clientctx->device_info;
    int &cam_count = *g_clientctx->cam_count;
    auto &check = *g_clientctx->check;

    int &num_cams = *g_clientctx->num_cameras;
    CameraParams *&params = *g_clientctx->cameras_params;
    CameraEachSelect *&select = *g_clientctx->cameras_select;
    CameraEmergent *&ecams = *g_clientctx->ecams;
    ScrollingBuffer *&plots = *g_clientctx->realtime_plot_data;
    CameraControl *camera_control = g_clientctx->camera_control;

    std::vector<std::string> cfg_files;
    update_camera_configs(cfg_files, selected_folder);
    select_cameras_have_configs(cfg_files, devs, check, cam_count);
    open_selected_cameras(check, cam_count, devs, cfg_files, num_cams, params,
                          select, ecams, plots);
    camera_control->open = true;
}

static void on_open_phase_complete() {
    // Optional: persist metadata, update UI, etc.
}

static void on_startthread_phase_start(std::string encoder_setup,
                                       std::string folder_name) {
    // unpack
    std::thread &detection3d_thread = *g_clientctx->detection3d_thread;
    std::string &calib_yaml_folder = *g_clientctx->calib_yaml_folder;
    PTPParams *ptp_params = g_clientctx->ptp_params;
    int &num_cameras = *g_clientctx->num_cameras;
    int &evt_buffer_size = *g_clientctx->evt_buffer_size;
    int &display_gpu_id = *g_clientctx->display_gpu_id;
    GL_Texture *&tex_gl = *g_clientctx->tex_gl;
    CameraParams *&cameras_params = *g_clientctx->cameras_params;
    CameraEachSelect *&cameras_select = *g_clientctx->cameras_select;
    CameraEmergent *&ecams = *g_clientctx->ecams;
    std::vector<std::thread> *camera_threads = g_clientctx->camera_threads;

    // functionality
    make_folder(folder_name);
    ptp_params->network_sync = true;
    CameraControl *camera_control = g_clientctx->camera_control;
    camera_control->record_video = true;
    camera_control->subscribe = true;

    cudaSetDevice(display_gpu_id);
    tex_gl = new GL_Texture[num_cameras];
    for (int i = 0; i < num_cameras; i++) {
        if (cameras_select[i].stream_on) {
            int camera_width =
                int(cameras_params[i].width / cameras_select[i].downsample);
            int camera_height =
                int(cameras_params[i].height / cameras_select[i].downsample);
            setup_texture(tex_gl[i], camera_width, camera_height);
        }
    }

    start_camera_streaming(*camera_threads, camera_control, ecams,
                           cameras_params, cameras_select, tex_gl, num_cameras,
                           evt_buffer_size, true, encoder_setup, folder_name,
                           ptp_params, calib_yaml_folder, detection3d_thread);
}

static void on_startstreaming_phase_start(std::string folder_name,
                                          std::string save_format) {
    // unpack
    std::thread &detection3d_thread = *g_clientctx->detection3d_thread;
    std::string &calib_yaml_folder = *g_clientctx->calib_yaml_folder;
    PTPParams *ptp_params = g_clientctx->ptp_params;
    int &num_cameras = *g_clientctx->num_cameras;
    int &evt_buffer_size = *g_clientctx->evt_buffer_size;
    int &display_gpu_id = *g_clientctx->display_gpu_id;
    GL_Texture *&tex_gl = *g_clientctx->tex_gl;
    CameraParams *&cameras_params = *g_clientctx->cameras_params;
    CameraEachSelect *&cameras_select = *g_clientctx->cameras_select;
    CameraEmergent *&ecams = *g_clientctx->ecams;
    std::vector<std::thread> *camera_threads = g_clientctx->camera_threads;

    make_folder(folder_name);

    ptp_params->network_sync = false;
    CameraControl *camera_control = g_clientctx->camera_control;
    camera_control->record_video = false;
    camera_control->subscribe = true;

    cudaSetDevice(display_gpu_id);
    tex_gl = new GL_Texture[num_cameras];
    for (int i = 0; i < num_cameras; i++) {
        cameras_select[i].picture_save_folder = folder_name;
        cameras_select[i].frame_save_format = save_format;
        if (cameras_select[i].stream_on) {
            int camera_width =
                int(cameras_params[i].width / cameras_select[i].downsample);
            int camera_height =
                int(cameras_params[i].height / cameras_select[i].downsample);
            setup_texture(tex_gl[i], camera_width, camera_height);
        }
    }
    start_camera_streaming(*camera_threads, camera_control, ecams,
                           cameras_params, cameras_select, tex_gl, num_cameras,
                           evt_buffer_size, false, "", "", ptp_params,
                           calib_yaml_folder, detection3d_thread);
}

static void on_startrecord_phase_start(unsigned long long ptp_global_time) {
    PTPParams *ptp_params = g_clientctx->ptp_params;
    ptp_params->ptp_global_time = ptp_global_time;
    ptp_params->network_set_start_ptp = true;
}

static void on_stoprecord_phase_start(unsigned long long ptp_global_time) {
    PTPParams *ptp_params = g_clientctx->ptp_params;
    ptp_params->ptp_stop_time = ptp_global_time;
    ptp_params->network_set_stop_ptp = true;
}

static void on_takepicture_phase_start(uint picture_id) {
    CameraEachSelect *&cameras_select = *g_clientctx->cameras_select;
    int &num_cameras = *g_clientctx->num_cameras;

    for (int i = 0; i < num_cameras; i++) {
        cameras_select[i].pictures_counter = picture_id;
        cameras_select[i].frame_save_name =
            std::to_string(cameras_select[i].pictures_counter);
        cameras_select[i].sigs->frame_save_state.store(State_Copy_New_Frame);
    }
}

static void cleanup_host_client_resources() {
    PTPParams *ptp_params = g_clientctx->ptp_params;
    if (ptp_params->network_set_stop_ptp && ptp_params->ptp_stop_reached) {

        auto &check = *g_clientctx->check;
        int &cam_count = *g_clientctx->cam_count;
        int &num_cameras = *g_clientctx->num_cameras;
        CameraEachSelect *&cameras_select = *g_clientctx->cameras_select;
        CameraParams *&cameras_params = *g_clientctx->cameras_params;
        std::vector<std::thread> *camera_threads = g_clientctx->camera_threads;
        GL_Texture *&tex_gl = *g_clientctx->tex_gl;
        CameraControl *camera_control = g_clientctx->camera_control;
        int &evt_buffer_size = *g_clientctx->evt_buffer_size;
        CameraEmergent *&ecams = *g_clientctx->ecams;

        ptp_params->network_set_stop_ptp = false;

        for (int i = 0; i < num_cameras; i++) {
            if (cameras_select[i].stream_on) {
                int camera_width =
                    int(cameras_params[i].width / cameras_select[i].downsample);
                int camera_height = int(cameras_params[i].height /
                                        cameras_select[i].downsample);
                clear_upload_and_cleanup(tex_gl[i], camera_width,
                                         camera_height);
            }
        }
        delete[] tex_gl;
        tex_gl = nullptr;

        for (auto &t : *camera_threads)
            t.join();

        for (int i = 0; i < num_cameras; i++) {
            camera_threads->pop_back();
        }
        for (int i = 0; i < num_cameras; i++) {
            destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                                 evt_buffer_size, &cameras_params[i]);
            delete[] ecams[i].evt_frame;
            check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera),
                                cameras_params[i].camera_serial.c_str());
        }

        for (int i = 0; i < num_cameras; i++) {
            ptp_sync_off(&ecams[i].camera, &cameras_params[i]);
        }
        camera_control->sync_camera = false;
        camera_control->record_video = false;

        ptp_params->ptp_global_time = 0;
        ptp_params->ptp_stop_time = 0;
        ptp_params->ptp_counter = 0;
        ptp_params->ptp_stop_counter = 0;
        ptp_params->network_sync = false;
        ptp_params->network_set_start_ptp = false;
        ptp_params->ptp_stop_reached = false;
        ptp_params->ptp_start_reached = false;

        for (int i = 0; i < num_cameras; i++) {
            close_camera(&ecams[i].camera, &cameras_params[i]);
        }

        camera_control->open = false;

        for (int i = 0; i < cam_count; i++) {
            check[i] = 0;
        }
    }
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

static std::vector<uint8_t>
build_cmd_startstreaming(const std::string &job_id, uint32_t epoch,
                         uint32_t seq, const std::string &calib_folder,
                         const std::string &save_format) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(256);
    auto jid = b.CreateString(job_id);
    auto calib = b.CreateString(calib_folder);
    auto s_format = b.CreateString(save_format);
    auto args = CreateStartStreamingArgs(b, calib, s_format);
    auto msg = CreateServer(
        b, Kind_KindCommand, ServerControl_STARTSTREAMING, jid, epoch, seq,
        camnet::v1::CommandBody_StartStreamingArgs, args.Union(), 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t> build_cmd_bumblebeeboard(const std::string &job_id,
                                                     uint32_t epoch,
                                                     uint32_t seq) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(128);
    auto jid = b.CreateString(job_id);
    auto msg = CreateServer(b, Kind_KindCommand,
                            camnet::v1::ServerControl_BUMBLEBEEBOARD, jid,
                            epoch, seq, camnet::v1::CommandBody_NONE, 0, 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t> build_cmd_takepicture(const std::string &job_id,
                                                  uint32_t epoch, uint32_t seq,
                                                  uint picture_id) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(128);
    auto jid = b.CreateString(job_id);
    auto args = CreateTakePictureArgs(b, picture_id);
    auto msg =
        CreateServer(b, Kind_KindCommand, ServerControl_TAKEPICTURE, jid, epoch,
                     seq, CommandBody_TakePictureArgs, args.Union(), 0);
    b.Finish(msg);
    return {b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize()};
}

static std::vector<uint8_t> build_cmd_nextpose(const std::string &job_id,
                                               uint32_t epoch, uint32_t seq) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder b(128);
    auto jid = b.CreateString(job_id);
    auto msg =
        CreateServer(b, Kind_KindCommand, camnet::v1::ServerControl_NEXTPOSE,
                     jid, epoch, seq, camnet::v1::CommandBody_NONE, 0, 0);
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
    case camnet::v1::ServerControl_STARTSTREAMING:
        return "STARTSTREAMING";
    case camnet::v1::ServerControl_BUMBLEBEEBOARD:
        return "BUMBLEBEEBOARD";
    case camnet::v1::ServerControl_TAKEPICTURE:
        return "TAKEPICTURE";
    case camnet::v1::ServerControl_NEXTPOSE:
        return "NEXTPOSE";
    default:
        return "NONE";
    }
}

// ============================================================================
// Global host client state
// ============================================================================
static AppContext *g_ctxp = nullptr;

enum Phase {
    Phase_Open,
    Phase_Threads,
    Phase_Start,
    Phase_Stop,
    Phase_Done,
    Phase_Streaming,
    Phase_BumblebeeBoard,
    Phase_TakePicture,
    Phase_NextPose
};

static std::vector<std::pair<std::string, int>> g_endpoints; // host:port pairs
static std::vector<std::string> g_servers; // server names via bringup
static std::unordered_map<std::string, bool> g_ack_by;

static std::string g_jid = "recording";
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

static bool g_phase_started = false;

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
    case Phase_Streaming:
        return camnet::v1::ServerControl_STARTSTREAMING;
    case Phase_BumblebeeBoard:
        return camnet::v1::ServerControl_BUMBLEBEEBOARD;
    case Phase_TakePicture:
        return camnet::v1::ServerControl_TAKEPICTURE;
    case Phase_NextPose:
        return camnet::v1::ServerControl_NEXTPOSE;
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
    case Phase_Streaming:
        return "STREAMING";
    case Phase_BumblebeeBoard:
        return "BUMBLEBEEBOARD";
    case Phase_TakePicture:
        return "TAKEPICTURE";
    case Phase_NextPose:
        return "NEXTPOSE";
    default:
        return "?";
    }
}

static void advance_phase(std::string job_id) {

    if (job_id == "recording") {
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
    } else {
        // calibration
        switch (g_phase) {
        case Phase_Open:
            g_phase = Phase_Streaming;
            break;
        case Phase_Streaming:
            g_phase = Phase_BumblebeeBoard;
            break;
        case Phase_BumblebeeBoard:
            g_phase = Phase_TakePicture;
            break;
        case Phase_TakePicture:
            g_phase = Phase_NextPose;
            break;
        case Phase_NextPose:
            g_phase = Phase_TakePicture;
            break;
        default:
            break;
        }
    }
}

static void reset_session() {
    g_jid = "recording";
    g_epoch = 1;
    g_seq = 1;
    g_phase = Phase_Open;
    g_waiting = false;
    g_ack_by.clear();
    {
        std::lock_guard<std::mutex> lk(g_logs_m);
        g_logs.clear();
    }
    g_phase_started = false;
    g_folder_name = "";
    g_ptp_start_time = 0;
    g_ptp_stop_time = 0;
    g_picture_id = 0;
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
                               *g_clientctx->selected_network_folder);
        if (!g_phase_started) {
            on_open_phase_start();
            g_phase_started = true;
        }
        break;
    }
    case Phase_Threads: {
        std::string encoder_setup = "-codec " + *g_clientctx->encoder_codec +
                                    " -preset " + *g_clientctx->encoder_preset;
        if (!g_phase_started) {
            g_folder_name =
                *g_clientctx->input_folder + "/" + get_current_date_time();
        }
        bytes = build_cmd_startthreads(g_jid, g_epoch, g_seq, g_folder_name,
                                       encoder_setup);
        if (!g_phase_started) {
            on_startthread_phase_start(encoder_setup, g_folder_name);
            g_phase_started = true;
        }
        break;
    }
    case Phase_Start: {
        if (!g_phase_started) {
            CameraEmergent *&ecams = *g_clientctx->ecams;
            unsigned long long ptp_time =
                get_current_PTP_time(&ecams[0].camera);
            int delay_in_second = 3;
            g_ptp_start_time =
                ((unsigned long long)delay_in_second) * 1000000000 + ptp_time;
        }
        bytes = build_cmd_start(g_jid, g_epoch, g_seq, g_ptp_start_time);

        if (!g_phase_started) {
            on_startrecord_phase_start(g_ptp_start_time);
            g_phase_started = true;
        }

        break;
    }
    case Phase_Stop: {
        if (!g_phase_started) {
            CameraEmergent *&ecams = *g_clientctx->ecams;
            unsigned long long ptp_time =
                get_current_PTP_time(&ecams[0].camera);
            int delay_in_second = 3;
            g_ptp_stop_time =
                ((unsigned long long)delay_in_second) * 1000000000 + ptp_time;
        }

        bytes = build_cmd_stop(g_jid, g_epoch, g_seq, g_ptp_stop_time);
        if (!g_phase_started) {
            on_stoprecord_phase_start(g_ptp_stop_time);
            g_phase_started = true;
        }
        break;
    }
    case Phase_Streaming: {
        if (!g_phase_started) {
            g_folder_name =
                *g_clientctx->calib_save_folder + "/" + get_current_date_time();
        }
        std::string save_format = *g_clientctx->selected_picture_format;
        bytes = build_cmd_startstreaming(g_jid, g_epoch, g_seq, g_folder_name,
                                         save_format);

        if (!g_phase_started) {
            on_startstreaming_phase_start(g_folder_name, save_format);
            g_phase_started = true;
        }
        break;
    }
    case Phase_BumblebeeBoard: {
        if (!g_phase_started) {
            g_picture_id = 0;
        }
        bytes = build_cmd_bumblebeeboard(g_jid, g_epoch, g_seq);
        if (!g_phase_started) {
            g_phase_started = true;
        }
        break;
    }
    case Phase_TakePicture: {
        if (!g_phase_started) {
            g_picture_id++;
        }
        bytes = build_cmd_takepicture(g_jid, g_epoch, g_seq, g_picture_id);
        if (!g_phase_started) {
            on_takepicture_phase_start(g_picture_id);
            g_phase_started = true;
        }
        break;
    }
    case Phase_NextPose: {
        bytes = build_cmd_nextpose(g_jid, g_epoch, g_seq);
        if (!g_phase_started) {
            g_phase_started = true;
        }
        break;
    }
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

void host_client_start_net_thread(AppContext &ctx) {
    if (g_net_run.exchange(true))
        return;
    g_net_thr = std::thread(HostClient_NetThreadMain, &ctx);
}

void host_client_stop_net_thread() {
    if (!g_net_run.exchange(false))
        return;
    if (g_net_thr.joinable())
        g_net_thr.join();
}

// ============================================================================
// Public API
// ============================================================================
void host_client_set_step_in_tick(bool v) { g_step_net_in_tick = v; }

void host_client_init(
    AppContext &ctx,
    const std::vector<std::pair<std::string, int>> &endpoints) {
    g_ctxp = &ctx;
    g_endpoints = endpoints;

    const uint16_t listen_port = 3333;
    const size_t max_clients = 5;
    const size_t channels = 2;

    if (!g_ctxp->net.start_server(listen_port, max_clients, channels,
                                  /*in_bw=*/0, /*out_bw=*/0)) {
        std::fprintf(stderr, "[HOST] failed to listen on :%u (UDP)\n",
                     listen_port);
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

void host_client_tick() {
    if (g_step_net_in_tick) {
        (void)enet_dispatch_drain(g_ctxp->net, host_on_event);
    }

    update_servers_from_registry();

    // check if this server is ready
    bool this_server_ready = true;
    if (g_phase == Phase_TakePicture) {
        auto save_image_all_ready = *g_clientctx->save_image_all_ready;
        if (save_image_all_ready) {
            *g_clientctx->save_pics_counter = 0;
            // TODO: put this to on_phase_complete(); or have a global
            // this_server_ready flag
            // logf("This server ready.");
        } else {
            this_server_ready = false;
        }
    }

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
            if (all && this_server_ready) {
                logf("barrier OK for %s", ctrl_name(current_ctrl()));
                if (current_ctrl() == camnet::v1::ServerControl_OPENCAMERA) {
                    on_open_phase_complete();
                }
                g_phase_started = false;
                g_waiting = false;
                ++g_seq;
                advance_phase(g_jid);
            }
        }
    }

    cleanup_host_client_resources();
    // if (g_waiting) {
    //     auto now = std::chrono::steady_clock::now();
    //     if (now - g_last_send > std::chrono::milliseconds(g_timeout_ms)) {
    //         for (const auto &name : g_servers) {
    //             if (!g_ack_by.count(name) || !g_ack_by[name]) {
    //                 logf("waiting for %s on %s …", name.c_str(),
    //                      ctrl_name(current_ctrl()));
    //             }
    //         }
    //         broadcast_current_phase();
    //         g_last_send = std::chrono::steady_clock::now();
    //         logf("timeout → re-send %s", ctrl_name(current_ctrl()));
    //     }
    // }
}

void host_client_draw_gui() {
    ImGui::Begin("Network");

    std::string &selected_network_folder =
        *g_clientctx->selected_network_folder;
    int &network_config_select = *g_clientctx->network_config_select;
    auto &network_config_folders = *g_clientctx->network_config_folders;

    if (network_config_select < 0 ||
        network_config_select >= (int)network_config_folders.size()) {
        int idx =
            find_cfg_index(network_config_folders, selected_network_folder);
        network_config_select =
            (idx >= 0 ? idx : (network_config_folders.empty() ? -1 : 0));
        selected_network_folder = network_config_folders[network_config_select];
    }

    ImGuiStyle &style = ImGui::GetStyle();
    const int n = (int)network_config_folders.size();
    for (int i = 0; i < n; ++i) {
        if (i > 0)
            ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);

        std::string label = std::filesystem::path(network_config_folders[i])
                                .filename()
                                .string();

        const bool is_rig_new = (label == "rig_new");
        if (is_rig_new)
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(1.0f, 0.55f, 0.0f, 1.0f));

        if (ImGui::RadioButton((label + "##cfg" + std::to_string(i)).c_str(),
                               &network_config_select, i)) {
            selected_network_folder =
                network_config_folders[network_config_select];
        }

        if (is_rig_new)
            ImGui::PopStyleColor();
    }

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

    const char *options[] = {"recording", "calibration"};
    static int current = (g_jid == "recording") ? 0 : 1;
    ImGui::SetNextItemWidth(120.0f);

    if (ImGui::BeginCombo("Job Mode", options[current])) {
        for (int n = 0; n < IM_ARRAYSIZE(options); n++) {
            bool is_selected = (current == n);
            if (ImGui::Selectable(options[n], is_selected)) {
                current = n;
                g_jid = options[n];
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

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
