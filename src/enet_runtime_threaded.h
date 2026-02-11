#pragma once
#include "enet_types.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>

class EnetRuntime {
  public:
    ~EnetRuntime() { stop(); }

    // Server mode (bind & listen)
    bool start_server(uint16_t port, size_t max_clients = 64,
                      size_t channels = 2, enet_uint32 in_bw = 0,
                      enet_uint32 out_bw = 0);

    bool start_client(size_t channels = 2, enet_uint32 in_bw = 0,
                      enet_uint32 out_bw = 0);

    // Client connect (io thread will dial out)
    void connect(ConnectReq req) { connect_reqs_.push(std::move(req)); }

    // Optional graceful disconnect by peer id
    void disconnect(uint32_t peer_id, enet_uint32 data = 0);

    void stop();

    // App poll/produce APIs
    bool poll(Incoming &evt) { return incoming_.try_pop(evt); } // non-blocking
    void send(Outgoing msg) { outgoing_.push(std::move(msg)); } // thread-safe
    bool is_running() const { return running_.load(); }
    void peers_snapshot(std::vector<PeerSnapshot> &out);

  private:
    void io_loop();

    ENetHost *host_ = nullptr;
    std::thread io_thread_;
    std::atomic<bool> running_{false};

    TSQueue<Incoming> incoming_;
    TSQueue<Outgoing> outgoing_;
    TSQueue<ConnectReq> connect_reqs_;

    std::mutex peers_m_;
    std::unordered_map<uint32_t, ENetPeer *> peers_; // id -> peer*
    std::unordered_map<ENetPeer *, uint32_t> rev_;   // peer* -> id
    uint32_t next_peer_id_ = 1;
};
