#include "NvEncoder/NvCodecUtils.h"
#include "camera_manager.h"
#include "enet_fb_helpers.h"
#include "enet_runtime_select.h"
#include "fetch_generated.h" // your FB schema
#include <csignal>
#include <iostream>
#include <thread>

simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

static std::atomic<bool> g_quit{false};
extern "C" void on_sigint(int) {
    g_quit.store(true, std::memory_order_relaxed);
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
                    std::cout << "peer " << evt.peer_id << " connected\n";
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
                        mgr.state = ManagerState::OPENCAMERA;
                        break;

                    case FetchGame::ServerControl_STARTTHREAD:
                        record_folder = msg->record_folder()->str();
                        encoder_setup = msg->encoder_setup()->str();
                        mgr.state = ManagerState::STARTCAMTHREAD;
                        break;

                    case FetchGame::ServerControl_QUIT:
                        g_quit.store(true, std::memory_order_relaxed);
                        break;

                    case FetchGame::ServerControl_STARTRECORDING:
                        ptp.ptp_global_time = msg->ptp_global_time();
                        ptp.network_set_start_ptp = true;
                        mgr.state = ManagerState::WAITSTOP;
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
