#pragma once
#include "enet_fb_helpers.h" // EnetRuntime alias, PeerRegistry, fb_parse
#include "fetch_generated.h" // your FlatBuffers schema
#include <atomic>
#include <chrono>
#include <thread>

inline void run_enet_loop(EnetRuntime &net, PeerRegistry &peers,
                          std::atomic<bool> &stop) {
    // Common event handler used by both modes
    auto handle = [&](const Incoming &evt) {
        switch (evt.type) {
        case Incoming::Connect:
            peers.add(evt.peer_id);
            std::printf("[ENet] Peer %u connected\n", evt.peer_id);
            break;

        case Incoming::Disconnect:
            peers.remove(evt.peer_id);
            std::printf("[ENet] Peer %u disconnected\n", evt.peer_id);
            break;

        case Incoming::Receive: {
            const FetchGame::Server *msg = nullptr;
            const bool ok = fb_parse(
                evt.bytes,
                [](flatbuffers::Verifier &v) {
                    return FetchGame::VerifyServerBuffer(v);
                },
                [](const uint8_t *p) { return FetchGame::GetServer(p); }, msg);
            if (!ok || !msg) {
                std::fprintf(stderr, "[ENet] Dropped invalid packet from %u\n",
                             evt.peer_id);
                break;
            }

            switch (msg->signal_type()) {
            case FetchGame::SignalType_ClientBringup: {
                if (auto sm = msg->server_mesg()) {
                    if (auto n = sm->server_name())
                        peers.set_name(evt.peer_id, n->str());
                }

                break;
            }
            case FetchGame::SignalType_CalibrationPoseReached:
                std::puts("[ENet] Calibration pose reached.");
                break;

            case FetchGame::SignalType_CalibrationDone:
                std::puts("[ENet] Calibration done.");
                break;

            default:
                std::printf("[ENet] Peer %u state=%d\n", evt.peer_id,
                            msg->server_state());
                break;
            }
            break;
        }

        default:
            break;
        }
    };

#if defined(ENET_RUNTIME_THREADED)
    // Threaded: poll events produced by the I/O thread
    while (!stop.load(std::memory_order_relaxed)) {
        Incoming evt;
        while (net.poll(evt))
            handle(evt);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#else
    // Inline: service ENet and dispatch events via callback each frame
    while (!stop.load(std::memory_order_relaxed)) {
        net.step(2, handle); // 2 ms timeout; tweak as needed
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
}
