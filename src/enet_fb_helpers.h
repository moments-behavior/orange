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
        pid2cams_.erase(pid);
        cams_known_.erase(pid);
    }

    // ---- Name management ----
    void set_name(uint32_t pid, const std::string &name) {
        std::lock_guard<std::mutex> lk(m_);
        set_name_unlocked(pid, name);
    }
    std::string get_name(uint32_t pid) const {
        std::lock_guard<std::mutex> lk(m_);
        auto it = pid2name_.find(pid);
        return it == pid2name_.end() ? std::string() : it->second;
    }
    uint32_t get_pid_by_name(const std::string &name) const {
        std::lock_guard<std::mutex> lk(m_);
        auto it = name2pid_.find(name);
        return it == name2pid_.end() ? 0u : it->second;
    }

    // ---- Bringup for full servers (name + cameras + state are known) ----
    void set_bringup(uint32_t pid, const std::string &name, int cameras) {
        std::lock_guard<std::mutex> lk(m_);
        set_name_unlocked(pid, name);
        pid2cams_[pid] = cameras;
        cams_known_.insert(pid);
    }

    // Optional incremental updates:
    void set_cameras(uint32_t pid, int cameras) {
        std::lock_guard<std::mutex> lk(m_);
        pid2cams_[pid] = cameras;
        cams_known_.insert(pid);
    }
    void clear_cameras(uint32_t pid) {
        std::lock_guard<std::mutex> lk(m_);
        pid2cams_.erase(pid);
        cams_known_.erase(pid);
    }

    // Queries for GUI
    bool cameras_known(uint32_t pid, int *out = nullptr) const {
        std::lock_guard<std::mutex> lk(m_);
        if (cams_known_.count(pid) == 0)
            return false;
        if (out)
            *out = pid2cams_.at(pid);
        return true;
    }

    struct PeerInfo {
        uint32_t peer_id;
        std::string name;
        bool camsK;
        int cams;
    };
    std::vector<PeerInfo> snapshot_info() const {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<PeerInfo> out;
        out.reserve(peers_.size());
        for (auto pid : peers_) {
            auto n = pid2name_.find(pid);
            bool ck = cams_known_.count(pid) != 0;
            out.push_back({pid,
                           n == pid2name_.end() ? std::string() : n->second, ck,
                           ck ? pid2cams_.at(pid) : 0});
        }
        return out;
    }

  private:
    void set_name_unlocked(uint32_t pid, const std::string &name) {
        if (name.empty())
            return;
        if (auto it = pid2name_.find(pid);
            it != pid2name_.end() && it->second != name)
            name2pid_.erase(it->second);
        if (auto jt = name2pid_.find(name);
            jt != name2pid_.end() && jt->second != pid)
            pid2name_.erase(jt->second);
        pid2name_[pid] = name;
        name2pid_[name] = pid;
    }

    mutable std::mutex m_;
    std::unordered_set<uint32_t> peers_;
    std::unordered_map<uint32_t, std::string> pid2name_;
    std::unordered_map<std::string, uint32_t> name2pid_;

    std::unordered_map<uint32_t, int> pid2cams_;
    std::unordered_set<uint32_t> cams_known_;
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
