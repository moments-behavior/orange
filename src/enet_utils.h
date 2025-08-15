#pragma once
#include "enet_fb_helpers.h"
#include "fetch_game.h"

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

inline bool all_connected(const AppContext &ctx,
                          const std::vector<std::string> &server_names) {
    for (const auto &name : server_names)
        if (ctx.peers.get_pid_by_name(name) == 0)
            return false;
    return true;
}

inline void send_open_cameras_to(FBMessageSender &sender,
                                 const PeerRegistry &reg,
                                 const std::vector<std::string> &names,
                                 const std::string &config_folder) {
    auto build = [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        auto cfg = b.CreateString(config_folder);
        FetchGame::ServerBuilder sb(b);
        sb.add_config_folder(cfg);
        sb.add_control(FetchGame::ServerControl_OPENCAMERA);
        auto root = sb.Finish();
        b.Finish(root);
    };
    for (const auto &n : names) {
        if (uint32_t pid = reg.get_pid_by_name(n)) {
            sender.to_peer(pid, build);
        }
    }
}

inline void send_open_cameras_calib(FBMessageSender &sender,
                                    const PeerRegistry &reg,
                                    const std::vector<std::string> &names,
                                    const std::string &config_folder) {
    auto build = [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        auto cfg = b.CreateString(config_folder);
        FetchGame::ServerBuilder sb(b);
        sb.add_config_folder(cfg);
        sb.add_control(FetchGame::ServerControl_CALIBOPENCAMERA);
        auto root = sb.Finish();
        b.Finish(root);
    };
    for (const auto &n : names) {
        if (uint32_t pid = reg.get_pid_by_name(n)) {
            sender.to_peer(pid, build);
        }
    }
}

inline void send_start_threads_to(FBMessageSender &sender,
                                  const PeerRegistry &reg,
                                  const std::vector<std::string> &names,
                                  const std::string &record_folder_name,
                                  const std::string &encoder_basic_setup) {
    auto build = [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        auto rec = b.CreateString(record_folder_name);
        auto enc = b.CreateString(encoder_basic_setup);
        FetchGame::ServerBuilder sb(b);
        sb.add_record_folder(rec);
        sb.add_encoder_setup(enc);
        sb.add_control(FetchGame::ServerControl_STARTTHREAD);
        b.Finish(sb.Finish());
    };
    for (const auto &n : names)
        if (uint32_t pid = reg.get_pid_by_name(n))
            sender.to_peer(pid, build);
}

inline void send_start_calib(FBMessageSender &sender, const PeerRegistry &reg,
                             const std::vector<std::string> &names,
                             const std::string &folder_name) {
    auto build = [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        auto calib = b.CreateString(folder_name);
        FetchGame::ServerBuilder sb(b);
        sb.add_calib_folder(calib);
        sb.add_control(FetchGame::ServerControl_STARTCALIB);
        b.Finish(sb.Finish());
    };
    for (const auto &n : names)
        if (uint32_t pid = reg.get_pid_by_name(n))
            sender.to_peer(pid, build);
}

inline void send_set_start_ptp_to(FBMessageSender &sender,
                                  const PeerRegistry &reg,
                                  const std::vector<std::string> &names,
                                  std::uint64_t ptp_global_time) {
    auto build = [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        FetchGame::ServerBuilder sb(b);
        sb.add_control(FetchGame::ServerControl_STARTRECORDING);
        sb.add_ptp_global_time(ptp_global_time);
        b.Finish(sb.Finish());
    };
    for (const auto &n : names)
        if (uint32_t pid = reg.get_pid_by_name(n))
            sender.to_peer(pid, build);
}

inline void send_set_stop_ptp_to(FBMessageSender &sender,
                                 const PeerRegistry &reg,
                                 const std::vector<std::string> &names,
                                 std::uint64_t ptp_stop_time) {
    auto build = [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        FetchGame::ServerBuilder sb(b);
        sb.add_control(FetchGame::ServerControl_STOPRECORDING);
        sb.add_ptp_global_time(ptp_stop_time);
        b.Finish(sb.Finish());
    };
    for (const auto &n : names)
        if (uint32_t pid = reg.get_pid_by_name(n))
            sender.to_peer(pid, build);
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
