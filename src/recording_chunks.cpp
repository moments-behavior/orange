#include "recording_chunks.h"
#include "json.hpp"
#include "utils.h" // chown_to_invoking_user
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

std::string chunk_folder_name(const std::string &session_folder, int chunk) {
    char name[32];
    snprintf(name, sizeof(name), "chunk_%05d", chunk);
    return session_folder + "/" + name;
}

// ---------------------------------------------------------------- ChunkClock

void ChunkClock::configure(double chunk_seconds, int frame_rate) {
    reference_set = false;
    last_chunk = 0;
    ref_ns = 0;
    gate_offset_ns = 0;
    if (chunk_seconds <= 0.0 || frame_rate <= 0) {
        chunk_ns = 0;
        period_ns = 0;
        return;
    }
    chunk_ns = (long long)std::llround(chunk_seconds * 1e9);
    period_ns = (long long)std::llround(1e9 / frame_rate);
}

void ChunkClock::set_reference(unsigned long long first_ts,
                               unsigned long long gate_ptp_ns) {
    ref_ns = first_ts;
    gate_offset_ns = 0;
    if (gate_ptp_ns > 0 && period_ns > 0) {
        long long diff = (long long)(first_ts - gate_ptp_ns);
        gate_offset_ns = diff;
        // Snap to the gate's frame phase so a camera that lost its very first
        // frame still uses the same reference as the others. Only trusted when
        // the first frame arrived within a few seconds of the gate.
        if (std::llabs(diff) < 10LL * 1000000000LL) {
            long long n = (long long)std::llround((double)diff / period_ns);
            ref_ns = first_ts - (unsigned long long)(n * period_ns);
        }
    }
    reference_set = true;
}

int ChunkClock::chunk_of(unsigned long long ts) {
    if (chunk_ns <= 0)
        return 0;
    long long d = (long long)(ts - ref_ns) + period_ns / 2;
    int c = d < 0 ? 0 : (int)(d / chunk_ns);
    if (c < last_chunk)
        c = last_chunk;
    last_chunk = c;
    return c;
}

// ------------------------------------------------------------ chunk_registry

namespace chunk_registry {
namespace {
struct Session {
    int cameras = 0;
    std::map<int, std::vector<CameraChunkSummary>> chunks;
};
std::mutex g_mtx;
std::map<std::string, Session> g_sessions;

void write_chunk_complete(const std::string &session_folder, int chunk,
                          double chunk_seconds,
                          const std::vector<CameraChunkSummary> &cams) {
    using json = nlohmann::json;
    json j;
    j["chunk"] = chunk;
    j["chunk_seconds"] = chunk_seconds;
    j["session_folder"] = session_folder;
    j["written_unix_time"] =
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    unsigned long long ts_min = ~0ULL, ts_max = 0;
    unsigned long long frames_min = ~0ULL, frames_max = 0;
    bool all_key = true;
    json cj = json::array();
    for (const auto &c : cams) {
        ts_min = std::min(ts_min, c.first_timestamp);
        ts_max = std::max(ts_max, c.first_timestamp);
        frames_min = std::min(frames_min, c.frames);
        frames_max = std::max(frames_max, c.frames);
        all_key = all_key && c.first_packet_keyframe;
        cj.push_back({{"serial", c.camera_serial},
                      {"frames", c.frames},
                      {"metadata_rows", c.metadata_rows},
                      {"first_frame_id", c.first_frame_id},
                      {"last_frame_id", c.last_frame_id},
                      {"first_timestamp", c.first_timestamp},
                      {"last_timestamp", c.last_timestamp},
                      {"first_packet_keyframe", c.first_packet_keyframe},
                      {"finalize_ms", c.finalize_ms}});
    }
    j["cameras"] = cj;
    j["first_timestamp_spread_ns"] = cams.empty() ? 0 : ts_max - ts_min;
    j["frame_counts_equal"] = frames_min == frames_max;
    j["all_start_on_keyframe"] = all_key;

    std::string path = chunk_folder_name(session_folder, chunk) +
                       "/chunk_complete.json";
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        f << j.dump(2) << "\n";
    }
    std::rename(tmp.c_str(), path.c_str());
    chown_to_invoking_user(path);

    printf("[chunk] chunk %05d complete: %zu cameras, frames %llu-%llu, "
           "start spread %llu ns%s\n",
           chunk, cams.size(), frames_min, frames_max,
           cams.empty() ? 0ULL : ts_max - ts_min,
           all_key ? "" : ", NOT ALL START ON A KEYFRAME");
}
} // namespace

void register_camera(const std::string &session_folder) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_sessions[session_folder].cameras++;
}

void camera_finished_chunk(const std::string &session_folder, int chunk,
                           double chunk_seconds,
                           const CameraChunkSummary &summary) {
    std::vector<CameraChunkSummary> done;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_sessions.find(session_folder);
        if (it == g_sessions.end())
            return;
        auto &v = it->second.chunks[chunk];
        v.push_back(summary);
        if ((int)v.size() < it->second.cameras)
            return;
        done.swap(v);
        it->second.chunks.erase(chunk);
    }
    std::sort(done.begin(), done.end(),
              [](const CameraChunkSummary &a, const CameraChunkSummary &b) {
                  return a.camera_serial < b.camera_serial;
              });
    write_chunk_complete(session_folder, chunk, chunk_seconds, done);
}
} // namespace chunk_registry

// ------------------------------------------------------------ ChunkFinalizer

ChunkFinalizer::ChunkFinalizer() : worker(&ChunkFinalizer::run, this) {}

ChunkFinalizer::~ChunkFinalizer() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        stopping = true;
    }
    cv.notify_all();
    worker.join();
}

void ChunkFinalizer::post(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lk(mtx);
        jobs.push_back(std::move(job));
    }
    cv.notify_all();
}

size_t ChunkFinalizer::pending() {
    std::lock_guard<std::mutex> lk(mtx);
    return jobs.size() + (busy ? 1 : 0);
}

void ChunkFinalizer::run() {
    std::unique_lock<std::mutex> lk(mtx);
    for (;;) {
        cv.wait(lk, [this] { return stopping || !jobs.empty(); });
        if (jobs.empty()) {
            if (stopping)
                return;
            continue;
        }
        std::function<void()> job = std::move(jobs.front());
        jobs.pop_front();
        busy = true;
        lk.unlock();
        job();
        lk.lock();
        busy = false;
    }
}
