#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

namespace camera_timestamps {

enum class Format { None, OrangePTP };

struct CameraTimestamps {
    Format format = Format::None;
    std::map<std::string, std::vector<int64_t>> frame_ns;
    std::map<std::string, std::vector<int64_t>> frame_id;
    std::map<std::string, std::vector<int64_t>> frame_sys_ns;

    bool has(const std::string &cam) const {
        auto it = frame_ns.find(cam);
        return it != frame_ns.end() && !it->second.empty();
    }
};

namespace detail {

inline int64_t median_i64(std::vector<int64_t> v) {
    if (v.empty()) return 0;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

}

inline bool parse_orange_meta(const std::string &path,
                              std::vector<int64_t> &out_ns,
                              std::vector<int64_t> &out_id,
                              std::vector<int64_t> *out_sys = nullptr) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line)) return false;
    if (line.find("frame_id") == std::string::npos) return false;
    out_ns.clear();
    out_id.clear();
    if (out_sys) out_sys->clear();
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const char *b = line.c_str(), *e = b + line.size();
        int64_t fid = 0, ts = 0;
        auto r1 = std::from_chars(b, e, fid);
        if (r1.ec != std::errc() || r1.ptr >= e || *r1.ptr != ',') continue;
        auto r2 = std::from_chars(r1.ptr + 1, e, ts);
        if (r2.ec != std::errc()) continue;
        out_id.push_back(fid);
        out_ns.push_back(ts);
        if (out_sys) {
            int64_t sys = 0;
            if (r2.ptr < e && *r2.ptr == ',')
                std::from_chars(r2.ptr + 1, e, sys);
            out_sys->push_back(sys);
        }
    }
    return !out_ns.empty();
}

inline int64_t frame_period_ns(const std::vector<int64_t> &ns) {
    if (ns.size() < 2) return 0;
    std::vector<int64_t> d;
    d.reserve(ns.size() - 1);
    for (size_t i = 1; i < ns.size(); ++i) d.push_back(ns[i] - ns[i - 1]);
    return detail::median_i64(std::move(d));
}

inline std::vector<std::string> list_cameras(const std::string &folder) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    std::error_code ec;
    if (folder.empty() || !fs::is_directory(folder, ec)) return out;
    for (const auto &e : fs::directory_iterator(folder, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        const std::string suffix = "_meta.csv";
        if (name.rfind("Cam", 0) != 0) continue;
        if (name.size() <= 3 + suffix.size()) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        out.push_back(name.substr(3, name.size() - 3 - suffix.size()));
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline CameraTimestamps load(const std::string &folder,
                             const std::vector<std::string> &cam_ordered) {
    namespace fs = std::filesystem;
    CameraTimestamps out;
    std::error_code ec;
    if (folder.empty() || !fs::is_directory(folder, ec)) return out;

    std::map<std::string, std::vector<int64_t>> ns, ids, sys;
    int found = 0;
    for (const auto &cam : cam_ordered) {
        std::string p = (fs::path(folder) / ("Cam" + cam + "_meta.csv")).string();
        std::vector<int64_t> a, b, c;
        if (fs::exists(p, ec) && parse_orange_meta(p, a, b, &c)) {
            ns[cam] = std::move(a);
            ids[cam] = std::move(b);
            sys[cam] = std::move(c);
            ++found;
        }
    }
    if (found >= 2 || (found >= 1 && cam_ordered.size() == 1)) {
        out.format = Format::OrangePTP;
        out.frame_ns = std::move(ns);
        out.frame_id = std::move(ids);
        out.frame_sys_ns = std::move(sys);
    }
    return out;
}

}
