#include "camera_manager.h"
#include "enet_fb_helpers.h"
#include "enet_runtime_select.h"
#include "fetch_generated.h"
#include <csignal>
#include <iostream>
#include <thread>

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

int main(int, char **) {
    try {
        ENetGuard enet_guard;
        std::signal(SIGINT, on_sigint);

        EnetRuntime net; // alias -> EnetRuntimeInline here
        if (!net.start_server(3333)) {
            std::cerr << "ENet host create failed\n";
            return 1;
        }

        PeerRegistry peers;
        FBMessageSender sender{&net, /*channel=*/0, ENET_PACKET_FLAG_RELIABLE};

        // Camera manager
        int cam_count = 0;
        GigEVisionDeviceInfo unsorted[20]{}, sorted[20]{};
        std::string cfg_folder, record_folder = "recordings",
                                encoder_setup = "h264";
        PTPParams ptp;
        ManagerContext mgr;
        CameraManager cmgr;
        cmgr.start(&cam_count, &mgr, unsorted, sorted, &cfg_folder,
                   record_folder, encoder_setup, &ptp);

        while (!g_quit.load(std::memory_order_relaxed)) {
            // Service ENet and handle events inline
            net.step(2, [&](const Incoming &evt) {
                if (evt.type == Incoming::Connect) {
                    peers.add(evt.peer_id);
                    peers.set_name(evt.peer_id, "orange");
                    std::cout << "peer " << evt.peer_id << " connected\n";
                    if (mgr.state == FetchGame::ManagerState_IDLE) {
                        std::puts("Network: Successfully connected! Rescanning "
                                  "cameras.");
                        mgr.state = FetchGame::ManagerState_CONNECT;
                    } else {
                        std::puts("Network: Successfully connected!");
                        send_client_bringup(sender, evt.peer_id, cam_count,
                                            mgr.state);
                    }
                } else if (evt.type == Incoming::Disconnect) {
                    peers.remove(evt.peer_id);
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
                        cfg_folder = msg->config_folder()->str();
                        mgr.state = FetchGame::ManagerState_OPENCAMERA;
                        break;

                    case FetchGame::ServerControl_STARTTHREAD:
                        record_folder = msg->record_folder()->str();
                        encoder_setup = msg->encoder_setup()->str();
                        mgr.state = FetchGame::ManagerState_STARTCAMTHREAD;
                        break;

                    case FetchGame::ServerControl_QUIT:
                        g_quit.store(true, std::memory_order_relaxed);
                        break;

                    case FetchGame::ServerControl_STARTRECORDING:
                        ptp.ptp_global_time = msg->ptp_global_time();
                        ptp.network_set_start_ptp = true;
                        mgr.state = FetchGame::ManagerState_WAITSTOP;
                        // // Example reply (same thread; safe for inline send)
                        // sender.to_peer(
                        //     evt.peer_id,
                        //     [&](flatbuffers::FlatBufferBuilder &fbb) {
                        //         auto root = FetchGame::CreateServer(
                        //             fbb, FetchGame::SignalType_ServerState,
                        //             /*state=*/(int)ManagerState::WAITSTOP,
                        //             0);
                        //         fbb.Finish(root);
                        //     });
                        break;

                    case FetchGame::ServerControl_STOPRECORDING:
                        ptp.ptp_stop_time = msg->ptp_global_time();
                        ptp.network_set_stop_ptp = true;
                        break;

                    default:
                        break;
                    }
                }
            });

            // Example: do other per-frame work here
            // e.g., periodic broadcast using sender.broadcast(...)

            if (mgr.state == FetchGame::ManagerState_CONNECTED) {
                std::cout << "here?" << std::endl;
                mgr.state = FetchGame::ManagerState_IDLE;
                send_client_bringup(sender, peers.get_pid_by_name("orange"),
                                    cam_count, mgr.state);
            }
            if (mgr.state == FetchGame::ManagerState_CAMERAOPENED) {
                mgr.state = FetchGame::ManagerState_WAITTHREAD;
                // client_send_state_update_message(
                //     &client, fb_builder, &client.m_pNetwork->peers[0],
                //     manager_context.state);
            } else if (mgr.state == FetchGame::ManagerState_THREADREADY) {
                mgr.state = FetchGame::ManagerState_WAITSTART;
                // client_send_state_update_message(
                //     &client, fb_builder, &client.m_pNetwork->peers[0],
                //     manager_context.state);
            } else if (mgr.state == FetchGame::ManagerState_RECORDSTOPPED) {
                mgr.state = FetchGame::ManagerState_IDLE;
                // client_send_state_update_message(
                //     &client, fb_builder, &client.m_pNetwork->peers[0],
                //     manager_context.state);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        mgr.quit = true;
        cmgr.stop();
        net.stop();
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}
