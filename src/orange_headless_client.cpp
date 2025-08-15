// main.cpp (excerpt showing how network talks to the manager)
#include "camera_manager.h"
#include "enet_fb_helpers.h"
#include "enet_runtime_select.h"
#include "fetch_game.h"
#include <csignal>
#include <iostream>

static std::atomic<bool> g_quit{false};
extern "C" void on_sigint(int) {
    g_quit.store(true, std::memory_order_relaxed);
}

int main() try {
    AppContext ctx;
    std::signal(SIGINT, on_sigint);
    if (!ctx.net.start_server(3333))
        return 1;

    CameraManager mgr(ctx, [&](const ManagerEvent &ev) {
        const auto pid = ctx.peers.get_pid_by_name("orange");

        switch (ev.state) {
        case FetchGame::ManagerState_CONNECTED:
            // Your old code sent bringup and set next state to IDLE.
            // We can send bringup with IDLE here (no need to mutate state
            // locally).
            send_client_bringup(ctx.sender, pid, mgr.cam_count(),
                                FetchGame::ManagerState_IDLE);
            break;

        case FetchGame::ManagerState_CAMERAOPENED:
            // You previously flipped to WAITTHREAD and sent it.
            send_client_state_update_message(
                ctx.sender, pid, FetchGame::ManagerState_CAMERAOPENED);
            break;
        case FetchGame::ManagerState_THREADREADY:
            // You previously flipped to WAITSTART and sent it.
            send_client_state_update_message(
                ctx.sender, pid, FetchGame::ManagerState_THREADREADY);
            break;

        case FetchGame::ManagerState_RECORDINGSTARTED:
            // You previously flipped to WAITSTART and sent it.
            send_client_state_update_message(
                ctx.sender, pid, FetchGame::ManagerState_RECORDINGSTARTED);
            break;
        case FetchGame::ManagerState_CALIBFOLDER:
            // You previously flipped back to IDLE and sent it.
            send_client_state_update_message(
                ctx.sender, pid, FetchGame::ManagerState_CALIBFOLDER);
            break;
        case FetchGame::ManagerState_CALIBCAMERAOPENED:
            // You previously flipped to WAITTHREAD and sent it.
            send_client_state_update_message(
                ctx.sender, pid, FetchGame::ManagerState_CALIBCAMERAOPENED);
            break;
        case FetchGame::ManagerState_RECORDSTOPPED:
            // You previously flipped back to IDLE and sent it.
            send_client_state_update_message(ctx.sender, pid,
                                             FetchGame::ManagerState_IDLE);
            break;

        default:
            break;
        }
    });

    // Kick off initial scan on first connect
    while (!g_quit.load(std::memory_order_relaxed)) {
        ctx.net.step(2, [&](const Incoming &evt) {
            if (evt.type == Incoming::Connect) {
                ctx.peers.add(evt.peer_id);
                ctx.peers.set_name(evt.peer_id, "orange");
                mgr.post({ManagerCmdType::ConnectScan});
            } else if (evt.type == Incoming::Disconnect) {
                ctx.peers.remove(evt.peer_id);
            } else if (evt.type == Incoming::Receive) {
                const FetchGame::Server *msg = nullptr;
                bool ok = fb_parse(
                    evt.bytes,
                    [](flatbuffers::Verifier &v) {
                        return FetchGame::VerifyServerBuffer(v);
                    },
                    [](const uint8_t *p) { return FetchGame::GetServer(p); },
                    msg);
                if (!ok || !msg)
                    return;

                switch (msg->control()) {
                case FetchGame::ServerControl_OPENCAMERA: {
                    OpenArgs o{};
                    o.folder = msg->config_folder()->str();
                    mgr.post(ManagerCmd{ManagerCmdType::OpenCameras, o});
                } break;
                case FetchGame::ServerControl_STARTTHREAD: {
                    StartArgs s{};
                    s.rec.record_folder = msg->record_folder()->str();
                    s.rec.encoder_basic_setup = msg->encoder_setup()->str();
                    static PTPParams ptp{0, 0, 0, 0, true, false, false, false};
                    s.ptp = &ptp;
                    mgr.post(ManagerCmd{ManagerCmdType::StartThreads, {}, s});
                } break;
                case FetchGame::ServerControl_STARTRECORDING:
                    mgr.post(ManagerCmd{ManagerCmdType::StartRecording,
                                        {},
                                        {},
                                        msg->ptp_global_time()});
                    break;
                case FetchGame::ServerControl_STARTCALIB: {
                    OpenArgs o{};
                    o.folder = msg->calib_folder()->str();
                    mgr.post(ManagerCmd{ManagerCmdType::StartCalib, o});
                } break;
                case FetchGame::ServerControl_CALIBOPENCAMERA: {
                    OpenArgs o{};
                    o.folder = msg->config_folder()->str();
                    mgr.post(ManagerCmd{ManagerCmdType::CalibOpenCameras, o});
                } break;
                case FetchGame::ServerControl_STOPRECORDING:
                    mgr.post(ManagerCmd{ManagerCmdType::StopRecording,
                                        {},
                                        {},
                                        msg->ptp_global_time()});
                    break;
                default:
                    break;
                }
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // graceful exit
    mgr.post({ManagerCmdType::Shutdown});
    ctx.net.stop();
    return 0;

} catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
}
