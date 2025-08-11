#include "enet_fb_helpers.h"
#include "enet_runtime_select.h"
#include "fetch_generated.h"
#include "utils.h"
#include "video_capture.h"
#include <csignal>
#include <iostream>
#include <thread>

#define evt_buffer_size 100
#define max_cameras 20

static std::atomic<bool> g_quit{false};
extern "C" void on_sigint(int) {
    g_quit.store(true, std::memory_order_relaxed);
}

inline void send_client_bringup(FBMessageSender &sender, uint32_t peer_id,
                                int cam_count,
                                FetchGame::ManagerState mgr_state) {
    sender.to_peer(peer_id, [&](flatbuffers::FlatBufferBuilder &b) {
        char hostname[128]{};
        (void)gethostname(hostname, sizeof(hostname));
        if (!hostname[0])
            std::snprintf(hostname, sizeof(hostname), "unknown");

        auto server_name = b.CreateString(hostname);
        // NOTE: keep your schema’s function name exactly (it looked like
        // Createbring_up_message)
        auto msg = FetchGame::Createbring_up_message(b, server_name, cam_count);

        FetchGame::ServerBuilder sb(b);
        sb.add_signal_type(FetchGame::SignalType_ClientBringup);
        sb.add_server_mesg(msg);
        sb.add_server_state(mgr_state); // cast if your schema expects int
        auto root = sb.Finish();
        b.Finish(root);
    });
}

inline void
send_client_state_update_message(FBMessageSender &sender, uint32_t peer_id,
                                 FetchGame::ManagerState server_state) {
    sender.to_peer(peer_id, [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        FetchGame::ServerBuilder sb(b);
        sb.add_signal_type(FetchGame::SignalType_ClientStateUpdate);
        sb.add_server_state(server_state); // cast if schema uses int
        b.Finish(sb.Finish());
    });
}

bool open_cameras(CameraParams *cameras_params, CameraEmergent *ecams,
                  CameraEachSelect *cameras_select,
                  GigEVisionDeviceInfo *device_info, int num_cameras,
                  std::string config_folder) {
    std::vector<std::string> camera_config_files;
    update_camera_configs(camera_config_files, config_folder);

    for (int i = 0; i < num_cameras; i++) {
        set_camera_params(&cameras_params[i], &cameras_select[i],
                          &device_info[i], camera_config_files, i, num_cameras);
        open_camera_with_params(&ecams[i].camera,
                                &device_info[cameras_params[i].camera_id],
                                &cameras_params[i]);
    }
    return true;
}

bool start_camera_thread(std::vector<std::thread> &camera_threads,
                         CameraParams *cameras_params, CameraEmergent *ecams,
                         CameraControl *camera_control,
                         CameraEachSelect *cameras_select,
                         GigEVisionDeviceInfo *device_info, int num_cameras,
                         PTPParams *ptp_params, std::string record_folder,
                         std::string encoder_basic_setup, AppContext &ctx) {
    std::cout << "start camera sthread..." << std::endl;
    try {
        allocate_camera_frame_buffers(ecams, cameras_params, evt_buffer_size,
                                      num_cameras);
    } catch (...) {
        std::cout << "Failed to start thread..." << std::endl;
        return false;
    }

    camera_control->record_video = true;
    camera_control->subscribe = true;
    camera_control->sync_camera = true;

    // Creating a directory to save recorded video;
    if (mkdir(record_folder.c_str(), 0777) == -1) {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return false;
    } else {
        std::cout << "Recorded video saves to : " << record_folder << std::endl;
    }

    for (int i = 0; i < num_cameras; i++) {
        ptp_camera_sync(&ecams[i].camera, &cameras_params[i]);
    }

    for (int i = 0; i < num_cameras; i++) {
        cameras_select->stream_on = false;
    }

    for (int i = 0; i < num_cameras; i++) {
        camera_threads.push_back(std::thread(
            &acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i],
            camera_control, nullptr, encoder_basic_setup, record_folder,
            ptp_params, std::ref(ctx)));
    }

    // wait for all camera ready
    while (ptp_params->ptp_counter != num_cameras) {
        usleep(10);
    }

    return true;
}

struct ManagerContext {
    FetchGame::ManagerState state;
    bool quit;
};

struct RecordingContext {
    std::string record_folder;
    std::string encoder_basic_setup;
};

void create_camera_manager(int *cam_count, ManagerContext *manager_context,
                           GigEVisionDeviceInfo *unsorted_device_info,
                           GigEVisionDeviceInfo *device_info,
                           std::string *config_folder,
                           RecordingContext *recording_setup,
                           PTPParams *ptp_params, AppContext &ctx) {
    CameraEmergent *ecams;
    CameraParams *cameras_params;
    std::vector<std::thread> camera_threads;
    CameraEachSelect *cameras_select;
    CameraControl *camera_control = new CameraControl;

    manager_context->state = FetchGame::ManagerState_IDLE;
    while (!manager_context->quit) {
        switch (manager_context->state) {
        case FetchGame::ManagerState_CONNECT:
            *cam_count = scan_cameras(max_cameras, unsorted_device_info);
            std::cout << *cam_count << std::endl;
            sort_cameras_ip(unsorted_device_info, device_info, *cam_count);
            manager_context->state = FetchGame::ManagerState_CONNECTED;
            break;
        case FetchGame::ManagerState_OPENCAMERA:
            ecams = new CameraEmergent[*cam_count];
            cameras_params = new CameraParams[*cam_count];
            cameras_select = new CameraEachSelect[*cam_count];
            if (open_cameras(cameras_params, ecams, cameras_select, device_info,
                             *cam_count, *config_folder)) {
                manager_context->state = FetchGame::ManagerState_CAMERAOPENED;
            } else {
                manager_context->state = FetchGame::ManagerState_ERROR;
            }
            break;
        case FetchGame::ManagerState_STARTCAMTHREAD:

            std::cout << recording_setup->encoder_basic_setup << std::endl;
            if (start_camera_thread(
                    camera_threads, cameras_params, ecams, camera_control,
                    cameras_select, device_info, *cam_count, ptp_params,
                    recording_setup->record_folder,
                    recording_setup->encoder_basic_setup, ctx)) {
                manager_context->state = FetchGame::ManagerState_THREADREADY;
            } else {
                manager_context->state = FetchGame::ManagerState_ERROR;
            }
            break;
        case FetchGame::ManagerState_ERROR:
            manager_context->quit = true;
            break;
        }

        if (ptp_params->network_set_stop_ptp && ptp_params->ptp_stop_reached) {
            ptp_params->network_set_stop_ptp = false;
            for (auto &t : camera_threads)
                t.join();

            for (int i = 0; i < *cam_count; i++) {
                camera_threads.pop_back();
            }

            for (int i = 0; i < *cam_count; i++) {
                ptp_sync_off(&ecams[i].camera, &cameras_params[i]);
            }
            ptp_params->ptp_global_time = 0;
            ptp_params->ptp_stop_time = 0;
            ptp_params->ptp_counter = 0;
            ptp_params->ptp_stop_counter = 0;
            ptp_params->network_sync = true;
            ptp_params->network_set_start_ptp = false;
            ptp_params->ptp_stop_reached = false;
            ptp_params->ptp_start_reached = false;
            camera_control->sync_camera = false;

            for (int i = 0; i < *cam_count; i++) {
                destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                                     evt_buffer_size, &cameras_params[i]);
                delete[] ecams[i].evt_frame;
                check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera),
                                    cameras_params[i].camera_serial.c_str());
                close_camera(&ecams[i].camera, &cameras_params[i]);
            }
            delete[] ecams;
            delete[] cameras_params;
            delete[] cameras_select;
            manager_context->state = FetchGame::ManagerState_RECORDSTOPPED;
        }
        usleep(1000);
    }
}

int main(int, char **) {
    try {

        AppContext ctx; // ENetGuard constructed here (enet_initialize)
        std::signal(SIGINT, on_sigint);
        if (!ctx.net.start_server(3333))
            return 1;

        int cam_count;
        GigEVisionDeviceInfo unsorted_device_info[max_cameras];
        cam_count = scan_cameras(max_cameras, unsorted_device_info);
        GigEVisionDeviceInfo device_info[max_cameras];
        sort_cameras_ip(unsorted_device_info, device_info, cam_count);
        std::cout << "available no of cameras: " << cam_count << std::endl;

        std::string config_folder;
        RecordingContext recording_setup;
        ManagerContext manager_context;
        PTPParams *ptp_params =
            new PTPParams{0, 0, 0, 0, true, false, false, false};

        std::thread *manager_thread = new std::thread(
            &create_camera_manager, &cam_count, &manager_context,
            unsorted_device_info, device_info, &config_folder, &recording_setup,
            ptp_params, std::ref(ctx));

        while (!g_quit.load(std::memory_order_relaxed)) {
            // Service ENet and handle events inline
            ctx.net.step(2, [&](const Incoming &evt) {
                if (evt.type == Incoming::Connect) {
                    ctx.peers.add(evt.peer_id);
                    ctx.peers.set_name(evt.peer_id, "orange");
                    std::cout << "peer " << evt.peer_id << " connected\n";
                    if (manager_context.state == FetchGame::ManagerState_IDLE) {
                        std::puts("Network: Successfully connected! Rescanning "
                                  "cameras.");
                        manager_context.state = FetchGame::ManagerState_CONNECT;
                    } else {
                        std::puts("Network: Successfully connected!");
                        send_client_bringup(ctx.sender, evt.peer_id, cam_count,
                                            manager_context.state);
                    }
                } else if (evt.type == Incoming::Disconnect) {
                    ctx.peers.remove(evt.peer_id);
                    std::cout << "peer " << evt.peer_id << " disconnected\n";
                } else if (evt.type == Incoming::Receive) {
                    const FetchGame::Server *msg = nullptr;
                    bool ok = fb_parse(
                        evt.bytes,
                        [](flatbuffers::Verifier &v) {
                            return FetchGame::VerifyServerBuffer(v);
                        },
                        [](const uint8_t *p) {
                            return FetchGame::GetServer(p);
                        },
                        msg);
                    if (!ok || !msg)
                        return;

                    switch (msg->control()) {
                    case FetchGame::ServerControl_OPENCAMERA:
                        config_folder = msg->config_folder()->str();
                        manager_context.state =
                            FetchGame::ManagerState_OPENCAMERA;
                        break;

                    case FetchGame::ServerControl_STARTTHREAD:
                        recording_setup.record_folder =
                            msg->record_folder()->str();
                        recording_setup.encoder_basic_setup =
                            msg->encoder_setup()->str();
                        manager_context.state =
                            FetchGame::ManagerState_STARTCAMTHREAD;

                        break;

                    case FetchGame::ServerControl_STARTRECORDING:
                        ptp_params->ptp_global_time = msg->ptp_global_time();
                        ptp_params->network_set_start_ptp = true;
                        manager_context.state =
                            FetchGame::ManagerState_WAITSTOP;
                        send_client_state_update_message(
                            ctx.sender, evt.peer_id, manager_context.state);
                        break;

                    case FetchGame::ServerControl_STOPRECORDING:
                        printf("stop signal\n");
                        std::cout << msg->ptp_global_time();
                        ptp_params->ptp_stop_time = msg->ptp_global_time();
                        ptp_params->network_set_stop_ptp = true;
                        break;

                    default:
                        break;
                    }
                }
            });

            // Example: do other per-frame work here
            // e.g., periodic broadcast using sender.broadcast(...)
            if (manager_context.state == FetchGame::ManagerState_CONNECTED) {
                manager_context.state = FetchGame::ManagerState_IDLE;
                send_client_bringup(ctx.sender,
                                    ctx.peers.get_pid_by_name("orange"),
                                    cam_count, manager_context.state);
            }
            if (manager_context.state == FetchGame::ManagerState_CAMERAOPENED) {
                manager_context.state = FetchGame::ManagerState_WAITTHREAD;
                send_client_state_update_message(
                    ctx.sender, ctx.peers.get_pid_by_name("orange"),
                    manager_context.state);
            } else if (manager_context.state ==
                       FetchGame::ManagerState_THREADREADY) {
                manager_context.state = FetchGame::ManagerState_WAITSTART;
                send_client_state_update_message(
                    ctx.sender, ctx.peers.get_pid_by_name("orange"),
                    manager_context.state);
            } else if (manager_context.state ==
                       FetchGame::ManagerState_RECORDSTOPPED) {
                manager_context.state = FetchGame::ManagerState_IDLE;
                send_client_state_update_message(
                    ctx.sender, ctx.peers.get_pid_by_name("orange"),
                    manager_context.state);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        manager_context.quit = true;
        manager_thread->join();
        ctx.net.stop();
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}
