#pragma once
#include "enet_runtime_select.h"
#include <flatbuffers/flatbuffers.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

class PeerRegistry {
  public:
    void add(uint32_t pid) {
        std::lock_guard<std::mutex> lk(m_);
        peers_.insert(pid);
    }

    void remove(uint32_t pid) {
        std::lock_guard<std::mutex> lk(m_);
        peers_.erase(pid);
        if (auto it = pid2name_.find(pid); it != pid2name_.end()) {
            name2pid_.erase(it->second);
            pid2name_.erase(it);
        }
    }

    // Set/rename peer's name. Enforces a bijection (name <-> pid).
    void set_name(uint32_t pid, const std::string &name) {
        std::lock_guard<std::mutex> lk(m_);
        if (name.empty()) {
            // treat empty as “clear name”
            if (auto it = pid2name_.find(pid); it != pid2name_.end()) {
                name2pid_.erase(it->second);
                pid2name_.erase(it);
            }
            return;
        }
        // If this pid had an old name, drop it.
        if (auto it = pid2name_.find(pid);
            it != pid2name_.end() && it->second != name) {
            name2pid_.erase(it->second);
        }
        // If another pid was using this name, drop that mapping.
        if (auto jt = name2pid_.find(name);
            jt != name2pid_.end() && jt->second != pid) {
            pid2name_.erase(jt->second);
        }
        pid2name_[pid] = name;
        name2pid_[name] = pid;
    }

    uint32_t get_pid_by_name(const std::string &name) const {
        std::lock_guard<std::mutex> lk(m_);
        auto it = name2pid_.find(name);
        return it == name2pid_.end() ? 0u : it->second;
    }

    std::string get_name(uint32_t pid) const {
        std::lock_guard<std::mutex> lk(m_);
        auto it = pid2name_.find(pid);
        return it == pid2name_.end() ? std::string() : it->second;
    }

    std::vector<uint32_t> snapshot_ids() const {
        std::lock_guard<std::mutex> lk(m_);
        return {peers_.begin(), peers_.end()};
    }

    struct PeerInfo {
        uint32_t peer_id;
        std::string name;
    };
    std::vector<PeerInfo> snapshot_info() const {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<PeerInfo> out;
        out.reserve(peers_.size());
        for (auto pid : peers_) {
            auto it = pid2name_.find(pid);
            out.push_back(
                {pid, it == pid2name_.end() ? std::string() : it->second});
        }
        return out;
    }

  private:
    mutable std::mutex m_;
    std::unordered_set<uint32_t> peers_;
    std::unordered_map<uint32_t, std::string> pid2name_;
    std::unordered_map<std::string, uint32_t> name2pid_;
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
