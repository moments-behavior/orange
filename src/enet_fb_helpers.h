#pragma once
#include "enet_runtime_select.h"
#include <flatbuffers/flatbuffers.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Thread-safe peer + role registry
class PeerRegistry {
  public:
    void add(uint32_t pid) {
        std::lock_guard<std::mutex> lk(m_);
        peers_.insert(pid);
    }
    void remove(uint32_t pid) {
        std::lock_guard<std::mutex> lk(m_);
        peers_.erase(pid);
        for (auto it = role2peer_.begin(); it != role2peer_.end();)
            it->second == pid ? it = role2peer_.erase(it) : ++it;
    }
    void set_role(const std::string &role, uint32_t pid) {
        std::lock_guard<std::mutex> lk(m_);
        role2peer_[role] = pid;
    }
    uint32_t get_role(const std::string &role) const {
        std::lock_guard<std::mutex> lk(m_);
        auto it = role2peer_.find(role);
        return it == role2peer_.end() ? 0u : it->second;
    }
    std::vector<uint32_t> snapshot() const {
        std::lock_guard<std::mutex> lk(m_);
        return {peers_.begin(), peers_.end()};
    }

  private:
    mutable std::mutex m_;
    std::unordered_set<uint32_t> peers_;
    std::unordered_map<std::string, uint32_t> role2peer_;
};

// FlatBuffers build-and-send helpers (work with either runtime alias)
struct FBMessageSender {
    EnetRuntime *rt = nullptr;
    uint8_t channel = 0;
    enet_uint32 flags = ENET_PACKET_FLAG_RELIABLE;

    template <class BuildFn> void to_peer(uint32_t pid, BuildFn &&build) const {
        if (!rt || !pid)
            return;
        thread_local flatbuffers::FlatBufferBuilder fbb(1024);
        fbb.Clear();
        build(fbb); // user calls Finish(root)
        auto *p = fbb.GetBufferPointer();
        auto n = fbb.GetSize();
        Outgoing o;
        o.peer_id = pid;
        o.channel = channel;
        o.flags = flags;
        o.bytes.assign(p, p + n);
        rt->send(o);
    }
    template <class BuildFn>
    void to_many(const std::vector<uint32_t> &ids, BuildFn &&build) const {
        if (!rt || ids.empty())
            return;
        thread_local flatbuffers::FlatBufferBuilder fbb(1024);
        fbb.Clear();
        build(fbb);
        auto *p = fbb.GetBufferPointer();
        auto n = fbb.GetSize();
        for (auto pid : ids) {
            if (!pid)
                continue;
            Outgoing o;
            o.peer_id = pid;
            o.channel = channel;
            o.flags = flags;
            o.bytes.assign(p, p + n);
            rt->send(o);
        }
    }
    template <class BuildFn> void broadcast(BuildFn &&build) const {
        if (!rt)
            return;
        thread_local flatbuffers::FlatBufferBuilder fbb(1024);
        fbb.Clear();
        build(fbb);
        auto *p = fbb.GetBufferPointer();
        auto n = fbb.GetSize();
        Outgoing o;
        o.peer_id = 0;
        o.channel = channel;
        o.flags = flags;
        o.bytes.assign(p, p + n);
        rt->send(o);
    }
    template <class BuildFn>
    void broadcast_except(const std::vector<uint32_t> &ids_all,
                          uint32_t except_id, BuildFn &&build) const {
        if (!rt)
            return;
        thread_local flatbuffers::FlatBufferBuilder fbb(1024);
        fbb.Clear();
        build(fbb);
        auto *p = fbb.GetBufferPointer();
        auto n = fbb.GetSize();
        for (auto pid : ids_all) {
            if (!pid || pid == except_id)
                continue;
            Outgoing o;
            o.peer_id = pid;
            o.channel = channel;
            o.flags = flags;
            o.bytes.assign(p, p + n);
            rt->send(o);
        }
    }
};

// Verify + parse incoming FB buffer
template <class VerifyFn, class GetFn, class RootPtr>
inline bool fb_parse(const std::vector<uint8_t> &bytes, VerifyFn &&verify_fn,
                     GetFn &&get_fn, RootPtr &out_root) {
    out_root = nullptr;
    if (bytes.empty())
        return false;
    flatbuffers::Verifier v(bytes.data(), bytes.size());
    if (!verify_fn(v))
        return false;
    out_root = get_fn(bytes.data());
    return out_root != nullptr;
}
