#pragma once
#include <cstdint>
#include <unordered_map>
extern "C" {
#include <enet/enet.h>
}

#include "enet_types.h" // Incoming, Outgoing, ConnectReq

// Queue-less, single-thread inline runtime.
// NOTE: Caller must ensure all calls happen on the same thread.
class EnetRuntimeInline {
  public:
    EnetRuntimeInline() = default;

    // Server mode (bind & listen)
    bool start_server(uint16_t port, size_t max_clients = 64,
                      size_t channels = 2, enet_uint32 in_bw = 0,
                      enet_uint32 out_bw = 0) {
        if (host_)
            return true;
        ENetAddress addr;
        addr.host = ENET_HOST_ANY;
        addr.port = port;
        host_ = enet_host_create(&addr, max_clients, channels, in_bw, out_bw);
        if (!host_)
            return false;
        channels_ = channels;
        return true;
    }

    // Client connect (creates client host lazily if needed)
    bool connect(const ConnectReq &cr) {
        if (!host_) {
            host_ =
                enet_host_create(nullptr, /*max peers*/ 8, cr.channels, 0, 0);
            if (!host_)
                return false;
            channels_ = cr.channels;
        }
        ENetAddress addr{};
        if (enet_address_set_host_ip(&addr, cr.host.c_str()) != 0) {
            if (enet_address_set_host(&addr, cr.host.c_str()) != 0)
                return false;
        }
        addr.port = cr.port;
        return enet_host_connect(host_, &addr, cr.channels, cr.user_data) !=
               nullptr;
    }

    // Step once. Call every frame. Invokes on_event for each ENet event.
    template <class EventFn> void step(int timeout_ms, EventFn &&on_event) {
        if (!host_)
            return;

        ENetEvent ev;
        while (enet_host_service(host_, &ev, timeout_ms) > 0) {
            if (ev.type == ENET_EVENT_TYPE_CONNECT) {
                const uint32_t pid = next_peer_id_++;
                peers_[pid] = ev.peer;
                rev_[ev.peer] = pid;
                on_event(Incoming{Incoming::Connect, pid, {}});
            } else if (ev.type == ENET_EVENT_TYPE_RECEIVE) {
                auto it = rev_.find(ev.peer);
                if (it != rev_.end()) {
                    Incoming inc;
                    inc.type = Incoming::Receive;
                    inc.peer_id = it->second;
                    inc.bytes.assign(ev.packet->data,
                                     ev.packet->data + ev.packet->dataLength);
                    enet_packet_destroy(ev.packet);
                    on_event(std::move(inc));
                } else {
                    enet_packet_destroy(ev.packet);
                }
                timeout_ms = 0; // subsequent polls non-blocking this tick
            } else if (ev.type == ENET_EVENT_TYPE_DISCONNECT) {
                auto it = rev_.find(ev.peer);
                if (it != rev_.end()) {
                    const uint32_t pid = it->second;
                    rev_.erase(it);
                    peers_.erase(pid);
                    on_event(Incoming{Incoming::Disconnect, pid, {}});
                }
            }
            timeout_ms = 0;
        }
    }

    // Immediate send (same thread as step/start/connect/stop)
    void send(const Outgoing &out) {
        if (!host_)
            return;
        ENetPacket *pkt =
            enet_packet_create(out.bytes.data(), out.bytes.size(), out.flags);
        if (!pkt)
            return;
        if (out.peer_id == 0) {
            enet_host_broadcast(host_, out.channel, pkt);
        } else {
            auto it = peers_.find(out.peer_id);
            if (it != peers_.end())
                enet_peer_send(it->second, out.channel, pkt);
            else
                enet_packet_destroy(pkt);
        }
        enet_host_flush(host_);
    }

    void disconnect(uint32_t peer_id, enet_uint32 data = 0) {
        auto it = peers_.find(peer_id);
        if (it != peers_.end())
            enet_peer_disconnect(it->second, data);
    }

    void stop() {
        if (!host_)
            return;
        enet_host_destroy(host_);
        host_ = nullptr;
        peers_.clear();
        rev_.clear();
        next_peer_id_ = 1;
        channels_ = 0;
    }

  private:
    ENetHost *host_ = nullptr;
    size_t channels_ = 0;

    std::unordered_map<uint32_t, ENetPeer *> peers_; // id -> peer*
    std::unordered_map<ENetPeer *, uint32_t> rev_;   // peer* -> id
    uint32_t next_peer_id_ = 1;
};
