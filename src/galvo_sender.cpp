#include "galvo_sender.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

#pragma pack(push, 1)
struct GcTargetPacket {
    char magic[4];    // 'G','C','T','1'
    uint16_t version; // 1
    uint16_t flags;   // bit0 = valid
    uint32_t seq;
    uint32_t target_id;
    uint64_t timestamp_ns;
    double x, y, z; // world mm
};

// v2 adds velocity + measured capture->send age so the receiver can
// extrapolate the target to its own aim time (see PROTOCOL.md).
struct GcTargetPacketV2 {
    char magic[4];    // 'G','C','T','1'
    uint16_t version; // 2
    uint16_t flags;   // bit0 = valid
    uint32_t seq;
    uint32_t target_id;
    uint64_t timestamp_ns;
    double x, y, z;    // world mm, at CAPTURE time
    double vx, vy, vz; // world mm/s
    uint32_t age_us;   // capture -> send latency measured by the sender
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(GcTargetPacket) == 48,
             "GcTargetPacket layout must match the 48-byte GCT1 wire format");
static_assert(sizeof(GcTargetPacketV2) == 80,
             "GcTargetPacketV2 layout must match the 80-byte GCT1 v2 wire format");

std::mutex g_mutex;
int g_sock_fd = -1;
sockaddr_in g_dst{};
std::atomic<uint32_t> g_seq{0};

std::thread g_dummy_thread;
std::atomic<bool> g_dummy_running{false};

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
          static_cast<uint64_t>(ts.tv_nsec);
}

} // namespace

bool galvo_sender_start(const std::string &target_ip, int target_port) {
    std::lock_guard<std::mutex> lock(g_mutex);

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::cerr << "galvo_sender: failed to create UDP socket" << std::endl;
        return false;
    }

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(static_cast<uint16_t>(target_port));
    if (inet_pton(AF_INET, target_ip.c_str(), &dst.sin_addr) != 1) {
        std::cerr << "galvo_sender: invalid target IP '" << target_ip << "'"
                  << std::endl;
        close(fd);
        return false;
    }

    if (g_sock_fd >= 0) {
        close(g_sock_fd);
    }
    g_sock_fd = fd;
    g_dst = dst;
    g_seq.store(0);
    return true;
}

void galvo_sender_stop() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_sock_fd >= 0) {
        close(g_sock_fd);
        g_sock_fd = -1;
    }
}

bool galvo_sender_running() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_sock_fd >= 0;
}

void galvo_sender_send_target(double x, double y, double z, bool valid,
                              uint32_t target_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_sock_fd < 0) {
        return;
    }

    GcTargetPacket pkt;
    std::memcpy(pkt.magic, "GCT1", 4);
    pkt.version = 1;
    pkt.flags = valid ? 1 : 0;
    pkt.seq = g_seq.fetch_add(1);
    pkt.target_id = target_id;
    pkt.timestamp_ns = now_ns();
    pkt.x = x;
    pkt.y = y;
    pkt.z = z;

    // Fire-and-forget, no ACK, per protocol.
    sendto(g_sock_fd, &pkt, sizeof(pkt), 0,
          reinterpret_cast<sockaddr *>(&g_dst), sizeof(g_dst));
}

static void send_target_v2(double x, double y, double z, double vx, double vy,
                           double vz, bool valid, uint32_t age_us,
                           uint32_t target_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_sock_fd < 0) {
        return;
    }
    GcTargetPacketV2 pkt;
    std::memset(&pkt, 0, sizeof(pkt));
    std::memcpy(pkt.magic, "GCT1", 4);
    pkt.version = 2;
    pkt.flags = valid ? 1 : 0;
    pkt.seq = g_seq.fetch_add(1);
    pkt.target_id = target_id;
    pkt.timestamp_ns = now_ns();
    pkt.x = x;
    pkt.y = y;
    pkt.z = z;
    pkt.vx = vx;
    pkt.vy = vy;
    pkt.vz = vz;
    pkt.age_us = age_us;
    sendto(g_sock_fd, &pkt, sizeof(pkt), 0,
           reinterpret_cast<sockaddr *>(&g_dst), sizeof(g_dst));
}

void galvo_sender_send_target_tracked(double x, double y, double z, bool valid,
                                      uint64_t capture_mono_ns,
                                      uint32_t target_id) {
    // single-caller state (the 3D detection loop)
    static bool have_prev = false;
    static double px, py, pz, pt;
    static double vx = 0.0, vy = 0.0, vz = 0.0;

    if (!valid) {
        have_prev = false;
        vx = vy = vz = 0.0;
        send_target_v2(0, 0, 0, 0, 0, 0, false, 0, target_id);
        return;
    }

    // velocity from capture timestamps (frame-accurate dt); EMA keeps
    // triangulation jitter from being amplified by the receiver's
    // extrapolation
    double t = (capture_mono_ns != 0 ? capture_mono_ns : now_ns()) * 1e-9;
    if (have_prev) {
        double dt = t - pt;
        if (dt > 1e-4 && dt < 0.2) {
            constexpr double kAlpha = 0.25;
            vx += kAlpha * ((x - px) / dt - vx);
            vy += kAlpha * ((y - py) / dt - vy);
            vz += kAlpha * ((z - pz) / dt - vz);
        } else {
            vx = vy = vz = 0.0; // stale gap: don't trust the old velocity
        }
    }
    px = x; py = y; pz = z; pt = t;
    have_prev = true;

    uint64_t now = now_ns();
    uint32_t age_us = 0;
    if (capture_mono_ns != 0 && now > capture_mono_ns) {
        uint64_t age = (now - capture_mono_ns) / 1000ull;
        age_us = age > 500000ull ? 500000u : static_cast<uint32_t>(age);
    }
    send_target_v2(x, y, z, vx, vy, vz, true, age_us, target_id);
}

void galvo_sender_start_dummy_target() {
    if (g_dummy_running.exchange(true)) {
        return; // already running
    }

    g_dummy_thread = std::thread([] {
        // Same test-circle pattern the GalvoControl receiver traces itself
        // when idle: base (0,0,0) + 300mm radius circle in x/y, 1000mm in z.
        constexpr double kRadiusMm = 300.0;
        constexpr double kZOffsetMm = 1000.0;
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        while (g_dummy_running.load()) {
            double t = std::chrono::duration<double>(Clock::now() - t0).count();
            double x = kRadiusMm * std::cos(t);
            double y = kRadiusMm * std::sin(t);
            // v2 with the analytic velocity: exercises the receiver's
            // extrapolation + feed-forward path without cameras
            send_target_v2(x, y, kZOffsetMm, -kRadiusMm * std::sin(t),
                           kRadiusMm * std::cos(t), 0.0, true, 0, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); // ~200 Hz
        }
    });
}

void galvo_sender_stop_dummy_target() {
    if (!g_dummy_running.exchange(false)) {
        return; // wasn't running
    }
    if (g_dummy_thread.joinable()) {
        g_dummy_thread.join();
    }
}

bool galvo_sender_dummy_target_running() { return g_dummy_running.load(); }
