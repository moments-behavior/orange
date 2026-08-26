#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "camera_timestamps.h"
#include "json.hpp"

namespace sync_plan {

constexpr int PLAN_VERSION = 1;
constexpr double DROP_FACTOR = 1.5;

struct Gap {
    int64_t slot = 0;
    int64_t lost = 0;
    int64_t pos_before = 0;
};

struct SyncCam {
    int64_t start_slot = 0;
    int64_t decoded_len = 0;
    int64_t end_slot = 0;
    std::vector<Gap> gaps;
    std::string frame_id_mode;

    void finalize() {
        std::sort(gaps.begin(), gaps.end(),
                  [](const Gap &a, const Gap &b) { return a.slot < b.slot; });
        int64_t total_lost = 0;
        for (auto &g : gaps) {
            g.pos_before = (g.slot - start_slot) - total_lost;
            total_lost += g.lost;
        }
        end_slot = start_slot + decoded_len + total_lost;
    }

    int64_t true_span() const { return end_slot - start_slot; }

    bool has(int64_t t) const {
        if (t < start_slot || t >= end_slot) return false;
        for (const auto &g : gaps) {
            if (g.slot <= t && t < g.slot + g.lost) return false;
            if (g.slot > t) break;
        }
        return true;
    }

    int64_t pos(int64_t t) const {
        int64_t lost_before = 0;
        for (const auto &g : gaps) {
            if (g.slot >= t) break;
            lost_before += g.lost;
        }
        return (t - start_slot) - lost_before;
    }

    int64_t slot_of_pos(int64_t p) const {
        int64_t lost = 0;
        for (const auto &g : gaps) {
            if (g.pos_before > p) break;
            lost += g.lost;
        }
        return start_slot + p + lost;
    }

    int64_t seek_pos(int64_t t) const {
        if (t <= start_slot) return 0;
        if (t >= end_slot) return decoded_len - 1;
        int64_t absent = 0;
        for (const auto &g : gaps) {
            if (g.slot >= t) break;
            absent += std::min(g.lost, t - g.slot);
        }
        return (t - start_slot) - absent;
    }
};

enum class Status { Clean, Trim, Reindex };

inline const char *status_name(Status s) {
    switch (s) {
        case Status::Clean: return "clean";
        case Status::Trim: return "trim";
        case Status::Reindex: return "reindex";
    }
    return "?";
}

struct SyncPlan {
    bool valid = false;
    std::string error;
    std::string recording;

    int64_t delta_ns = 0;
    int64_t anchor_ts_ns = 0;
    int64_t canonical_len = 0;
    int64_t predict_start = 0;
    int64_t predict_len = 0;
    int64_t first_drop_slot = -1;
    Status status = Status::Clean;
    std::map<std::string, SyncCam> cams;

    const SyncCam *cam(const std::string &name) const {
        auto it = cams.find(name);
        return it == cams.end() ? nullptr : &it->second;
    }
    bool usable() const { return valid && !cams.empty(); }
};

namespace detail {

inline std::string cam_token(const std::string &full_name) {
    if (full_name.rfind("Cam", 0) == 0 && full_name.size() > 3)
        return full_name.substr(3);
    return full_name;
}

}

inline SyncPlan build(const camera_timestamps::CameraTimestamps &ts,
                      const std::vector<std::string> &full_cam_names,
                      int64_t delta_ns_override = 0) {
    SyncPlan plan;
    if (full_cam_names.empty()) {
        plan.error = "no cameras";
        return plan;
    }

    std::map<std::string, const std::vector<int64_t> *> series;
    std::map<std::string, const std::vector<int64_t> *> ids;
    for (const auto &name : full_cam_names) {
        const std::string token = detail::cam_token(name);
        const std::vector<int64_t> *s = nullptr;
        const std::vector<int64_t> *fid = nullptr;
        if (ts.has(token)) {
            s = &ts.frame_ns.at(token);
            if (ts.frame_id.count(token)) fid = &ts.frame_id.at(token);
        } else if (ts.has(name)) {
            s = &ts.frame_ns.at(name);
            if (ts.frame_id.count(name)) fid = &ts.frame_id.at(name);
        }
        if (!s || s->size() < 2) {
            plan.error = "no timestamps for camera " + name;
            return plan;
        }
        series[name] = s;
        ids[name] = fid;
    }

    if (delta_ns_override > 0) {
        plan.delta_ns = delta_ns_override;
    } else {
        std::vector<int64_t> per_cam;
        for (const auto &kv : series) {
            int64_t p = camera_timestamps::frame_period_ns(*kv.second);
            if (p > 0) per_cam.push_back(p);
        }
        if (per_cam.empty()) {
            plan.error = "cannot infer frame interval";
            return plan;
        }
        plan.delta_ns = camera_timestamps::detail::median_i64(std::move(per_cam));
    }
    if (plan.delta_ns <= 0) {
        plan.error = "bad frame interval";
        return plan;
    }
    const double delta = (double)plan.delta_ns;

    int64_t anchor = INT64_MAX;
    for (const auto &kv : series) anchor = std::min(anchor, kv.second->front());
    plan.anchor_ts_ns = anchor;

    const int64_t thr = (int64_t)std::llround(DROP_FACTOR * delta);
    bool any_drops = false, starts_zero = true, ends_equal = true;
    int64_t first_end = -1;

    for (const auto &name : full_cam_names) {
        const auto &ns = *series[name];
        SyncCam c;
        c.decoded_len = (int64_t)ns.size();
        c.start_slot = (int64_t)std::llround((double)(ns.front() - anchor) / delta);

        int64_t cum_lost = 0;
        for (size_t i = 1; i < ns.size(); ++i) {
            int64_t d = ns[i] - ns[i - 1];
            if (d < thr) continue;
            int64_t lost = (int64_t)std::llround((double)d / delta) - 1;
            if (lost <= 0) continue;
            Gap g;
            g.lost = lost;
            g.slot = c.start_slot + (int64_t)i + cum_lost;
            c.gaps.push_back(g);
            cum_lost += lost;
        }
        c.finalize();

        const std::vector<int64_t> *fid = ids[name];
        if (fid && !fid->empty())
            c.frame_id_mode =
                (fid->back() - fid->front() == c.decoded_len - 1) ? "reindex" : "hole";

        int64_t true_span = c.decoded_len + cum_lost;
        int64_t ts_span =
            (int64_t)std::llround((double)(ns.back() - ns.front()) / delta) + 1;
        int64_t tol = std::max<int64_t>(3, (int64_t)(5e-4 * (double)true_span));
        if (std::llabs(ts_span - true_span) > tol) {
            plan.error = "timestamps inconsistent for camera " + name +
                         " (span " + std::to_string(ts_span) + " vs " +
                         std::to_string(true_span) + ")";
            return plan;
        }

        if (!c.gaps.empty()) any_drops = true;
        if (c.start_slot != 0) starts_zero = false;
        if (first_end < 0) first_end = c.end_slot;
        else if (c.end_slot != first_end) ends_equal = false;

        plan.canonical_len = std::max(plan.canonical_len, c.end_slot);
        plan.predict_start = std::max(plan.predict_start, c.start_slot);
        if (!c.gaps.empty())
            plan.first_drop_slot = (plan.first_drop_slot < 0)
                                       ? c.gaps.front().slot
                                       : std::min(plan.first_drop_slot,
                                                  c.gaps.front().slot);
        plan.cams[name] = std::move(c);
    }

    int64_t min_end = INT64_MAX;
    for (const auto &kv : plan.cams) min_end = std::min(min_end, kv.second.end_slot);
    plan.predict_len = std::max<int64_t>(0, min_end - plan.predict_start);

    plan.status = (!any_drops && starts_zero && ends_equal) ? Status::Clean
                  : !any_drops                              ? Status::Trim
                                                            : Status::Reindex;
    plan.valid = true;
    return plan;
}

inline bool self_check(const SyncPlan &plan, std::string *err = nullptr) {
    auto fail = [&](const std::string &m) {
        if (err) *err = m;
        return false;
    };
    for (const auto &kv : plan.cams) {
        const SyncCam &c = kv.second;
        int64_t expect_pos = 0;
        for (int64_t t = c.start_slot; t < c.end_slot; ++t) {
            if (!c.has(t)) continue;
            int64_t p = c.pos(t);
            if (p != expect_pos)
                return fail(kv.first + ": pos(" + std::to_string(t) + ")=" +
                            std::to_string(p) + " expected " +
                            std::to_string(expect_pos));
            if (c.slot_of_pos(p) != t)
                return fail(kv.first + ": slot_of_pos(" + std::to_string(p) +
                            ")=" + std::to_string(c.slot_of_pos(p)) +
                            " expected " + std::to_string(t));
            ++expect_pos;
        }
        if (expect_pos != c.decoded_len)
            return fail(kv.first + ": " + std::to_string(expect_pos) +
                        " present slots, decoded_len " +
                        std::to_string(c.decoded_len));
        if (c.has(c.start_slot - 1) || c.has(c.end_slot))
            return fail(kv.first + ": has() true outside span");
    }
    return true;
}

inline nlohmann::ordered_json to_json(const SyncPlan &plan) {
    nlohmann::ordered_json j;
    j["version"] = PLAN_VERSION;
    j["recording"] = plan.recording;
    j["delta_ns"] = plan.delta_ns;
    j["anchor_ts_ns"] = plan.anchor_ts_ns;
    j["canonical_len"] = plan.canonical_len;
    j["predict_start"] = plan.predict_start;
    j["predict_len"] = plan.predict_len;
    j["status"] = status_name(plan.status);
    if (plan.first_drop_slot >= 0) j["first_drop_slot"] = plan.first_drop_slot;
    else j["first_drop_slot"] = nullptr;

    nlohmann::ordered_json cams = nlohmann::ordered_json::object();
    for (const auto &kv : plan.cams) {
        const SyncCam &c = kv.second;
        nlohmann::ordered_json cj;
        cj["start_slot"] = c.start_slot;
        cj["decoded_len"] = c.decoded_len;
        cj["true_span"] = c.true_span();
        cj["frame_id_mode"] = c.frame_id_mode;
        nlohmann::ordered_json gaps = nlohmann::ordered_json::array();
        for (const auto &g : c.gaps)
            gaps.push_back({{"slot", g.slot}, {"lost", g.lost}});
        cj["gaps"] = gaps;
        cams[kv.first] = cj;
    }
    j["cameras"] = cams;
    return j;
}

inline bool write_json(const SyncPlan &plan, const std::string &path,
                       std::string *err = nullptr) {
    if (!plan.usable()) {
        if (err) *err = plan.error.empty() ? "plan is not usable" : plan.error;
        return false;
    }
    std::ofstream f(path);
    if (!f) {
        if (err) *err = "cannot write " + path;
        return false;
    }
    f << to_json(plan).dump();
    if (!f) {
        if (err) *err = "write failed: " + path;
        return false;
    }
    return true;
}

inline bool generate(const std::string &folder, std::string *err = nullptr) {
    namespace fs = std::filesystem;
    std::vector<std::string> tokens = camera_timestamps::list_cameras(folder);
    if (tokens.size() < 2) {
        if (err) *err = "no multi-camera metadata in " + folder;
        return false;
    }
    std::vector<std::string> names;
    names.reserve(tokens.size());
    for (const auto &t : tokens) names.push_back("Cam" + t);

    camera_timestamps::CameraTimestamps ts = camera_timestamps::load(folder, tokens);
    if (ts.format == camera_timestamps::Format::None) {
        if (err) *err = "could not read timestamps in " + folder;
        return false;
    }

    SyncPlan plan = build(ts, names);
    if (!plan.usable()) {
        if (err) *err = plan.error;
        return false;
    }
    std::string sc_err;
    if (!self_check(plan, &sc_err)) {
        if (err) *err = "plan self-check failed: " + sc_err;
        return false;
    }
    plan.recording = fs::path(folder).filename().string();
    if (plan.recording.empty())
        plan.recording = fs::path(folder).parent_path().filename().string();

    return write_json(plan, (fs::path(folder) / "sync_plan.json").string(), err);
}

}
