#include "enet_runtime_threaded.h"
extern "C" {
#include <enet/enet.h>
}

bool EnetRuntimeThreaded::start_server(uint16_t port, size_t max_clients,
                                       size_t channels, enet_uint32 in_bw,
                                       enet_uint32 out_bw) {
    if (running_)
        return true;
    ENetAddress addr;
    addr.host = ENET_HOST_ANY;
    addr.port = port;
    host_ = enet_host_create(&addr, max_clients, channels, in_bw, out_bw);
    if (!host_)
        return false;
    running_ = true;
    io_thread_ = std::thread(&EnetRuntimeThreaded::io_loop, this);
    return true;
}

void EnetRuntimeThreaded::disconnect(uint32_t peer_id, enet_uint32 data) {
    ENetPeer *p = nullptr;
    {
        std::lock_guard<std::mutex> lk(peers_m_);
        auto it = peers_.find(peer_id);
        if (it != peers_.end())
            p = it->second;
    }
    if (p)
        enet_peer_disconnect(p, data);
}

void EnetRuntimeThreaded::stop() {
    if (!running_.exchange(false))
        return;
    if (io_thread_.joinable())
        io_thread_.join();
    if (host_) {
        enet_host_destroy(host_);
        host_ = nullptr;
    }
    outgoing_.clear();
    connect_reqs_.clear();
    std::lock_guard<std::mutex> lk(peers_m_);
    peers_.clear();
    rev_.clear();
    next_peer_id_ = 1;
}

void EnetRuntimeThreaded::peers_snapshot(std::vector<PeerSnapshot> &out) {
    std::lock_guard<std::mutex> lk(peers_m_);
    out.clear();
    out.reserve(peers_.size());
    for (const auto &kv : peers_) {
        out.push_back(PeerSnapshot{kv.first, kv.second->address});
    }
}

void EnetRuntimeThreaded::io_loop() {
    while (running_.load()) {
        // Handle queued connect requests (client mode)
        ConnectReq cr;
        while (connect_reqs_.try_pop(cr)) {
            if (!host_) {
                host_ = enet_host_create(/*client*/ nullptr, /*max peers*/ 8,
                                         cr.channels, 0, 0);
                if (!host_)
                    continue;
            }
            ENetAddress addr{};
            if (enet_address_set_host_ip(&addr, cr.host.c_str()) != 0) {
                if (enet_address_set_host(&addr, cr.host.c_str()) != 0)
                    continue;
            }
            addr.port = cr.port;
            (void)enet_host_connect(host_, &addr, cr.channels, cr.user_data);
        }

        // Service network events
        ENetEvent ev;
        while (host_ && enet_host_service(host_, &ev, 2) > 0) {
            if (ev.type == ENET_EVENT_TYPE_CONNECT) {
                uint32_t id = next_peer_id_++;
                {
                    std::lock_guard<std::mutex> lk(peers_m_);
                    peers_[id] = ev.peer;
                    rev_[ev.peer] = id;
                }
                incoming_.push({Incoming::Connect, id, {}});
            } else if (ev.type == ENET_EVENT_TYPE_RECEIVE) {
                uint32_t id = 0;
                {
                    std::lock_guard<std::mutex> lk(peers_m_);
                    auto it = rev_.find(ev.peer);
                    if (it != rev_.end())
                        id = it->second;
                }
                if (id) {
                    Incoming inc;
                    inc.type = Incoming::Receive;
                    inc.peer_id = id;
                    inc.bytes.assign(ev.packet->data,
                                     ev.packet->data + ev.packet->dataLength);
                    enet_packet_destroy(ev.packet);
                    incoming_.push(std::move(inc));
                } else {
                    enet_packet_destroy(ev.packet);
                }
            } else if (ev.type == ENET_EVENT_TYPE_DISCONNECT) {
                uint32_t id = 0;
                {
                    std::lock_guard<std::mutex> lk(peers_m_);
                    auto it = rev_.find(ev.peer);
                    if (it != rev_.end()) {
                        id = it->second;
                        rev_.erase(it);
                        peers_.erase(id);
                    }
                }
                incoming_.push({Incoming::Disconnect, id, {}});
            }
        }

        // Drain outgoing
        Outgoing out;
        int sent = 0;
        while (outgoing_.try_pop(out)) {
            ENetPacket *pkt = enet_packet_create(out.bytes.data(),
                                                 out.bytes.size(), out.flags);
            if (!pkt)
                continue;
            if (out.peer_id == 0)
                enet_host_broadcast(host_, out.channel, pkt);
            else {
                ENetPeer *p = nullptr;
                {
                    std::lock_guard<std::mutex> lk(peers_m_);
                    auto it = peers_.find(out.peer_id);
                    if (it != peers_.end())
                        p = it->second;
                }
                if (p)
                    enet_peer_send(p, out.channel, pkt);
            }
            ++sent;
        }
        if (sent)
            enet_host_flush(host_);
    }
}
