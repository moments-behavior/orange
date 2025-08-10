#pragma once
#include <cstdint>
#include <enet/enet.h>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
extern "C" {
#include <enet/enet.h>
} // for enet_uint32
#include <stdexcept> // for std::runtime_error

// Event from ENet to the app
struct Incoming {
    enum Type { Connect, Receive, Disconnect } type;
    uint32_t peer_id = 0;
    std::vector<uint8_t> bytes; // set for Receive
};

// Packet from the app to ENet
struct Outgoing {
    uint32_t peer_id = 0; // 0 = broadcast
    uint8_t channel = 0;
    std::vector<uint8_t> bytes;
    enet_uint32 flags = ENET_PACKET_FLAG_RELIABLE;
};

// Client connect request
struct ConnectReq {
    std::string host; // IP or hostname
    uint16_t port = 0;
    size_t channels = 2;
    uint32_t user_data = 0; // optional cookie
};

// ENet init/deinit RAII
struct ENetGuard {
    ENetGuard() {
        if (enet_initialize() != 0)
            throw std::runtime_error("ENet init failed");
    }
    ~ENetGuard() { enet_deinitialize(); }
};

template <class T> class TSQueue {
  public:
    void push(T v) {
        std::lock_guard<std::mutex> lk(m_);
        q_.push(std::move(v));
    }
    bool try_pop(T &out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty())
            return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(m_);
        std::queue<T> empty;
        q_.swap(empty);
    }
    bool empty() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.empty();
    }

  private:
    mutable std::mutex m_;
    std::queue<T> q_;
};
