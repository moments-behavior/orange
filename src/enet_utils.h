#pragma once
#include "enet_fb_helpers.h"
#include "fetch_generated.h"

struct AppContext {
    ENetGuard guard; // must outlive everything that touches ENet
    EnetRuntime net;
    PeerRegistry peers;
    FBMessageSender sender{&net, /*channel=*/0, ENET_PACKET_FLAG_RELIABLE};

    AppContext() = default;
    AppContext(const AppContext &) = delete;
    AppContext &operator=(const AppContext &) = delete;
};

static void send_message(FBMessageSender &sender, uint32_t peer_id,
                         FetchGame::SignalType signal_type) {
    sender.to_peer(peer_id, [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        FetchGame::ServerBuilder sb(b);
        sb.add_signal_type(signal_type);
        b.Finish(sb.Finish());
    });
}

// Convenience: send to a peer by its name (e.g., "indigo")
inline void send_message_to_indigo(FBMessageSender &sender,
                                   const PeerRegistry &reg,
                                   const std::string &peer_name,
                                   FetchGame::SignalType signal_type) {
    if (uint32_t pid = reg.get_pid_by_name(peer_name)) {
        send_message(sender, pid, signal_type);
    }
}

inline bool all_in_state(const std::vector<std::string> &server_names,
                         const PeerRegistry &peers,
                         FetchGame::ManagerState want);

inline bool all_in_state(const std::vector<std::string> &server_names,
                         const PeerRegistry &peers,
                         FetchGame::ManagerState want) {
    for (const auto &name : server_names) {
        uint32_t pid = peers.get_pid_by_name(name);
        if (!pid)
            return false;

        int st = 0;
        if (!peers.state_known(pid, &st))
            return false;

        if (st != static_cast<int>(want))
            return false;
    }
    return true;
}
