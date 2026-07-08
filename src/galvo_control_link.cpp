#include "galvo_control_link.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

enum GcCommand : uint16_t {
    GC_CMD_PING = 0,
    GC_CMD_SET_ANGLES = 1,
    GC_CMD_CALIB_MODE = 2,
    GC_CMD_SET_CALIB = 3,
    GC_CMD_SAVE_CONFIG = 4,
    GC_CMD_STOP = 5,
};

#pragma pack(push, 1)
struct GcCommandPacket {
    char magic[4];    // 'G','C','C','1'
    uint16_t version; // 1
    uint16_t cmd;
    uint32_t seq;
    uint32_t reserved;
    double args[14];
};

struct GcStatusPacket {
    char magic[4];    // 'G','C','S','1'
    uint16_t version; // 1
    uint16_t cmd;     // echoed
    uint32_t seq;     // echoed
    uint32_t flags;   // bit0 ok, bit1 in_position, bit2 calib_mode,
                      // bit3 motors_enabled, bit4 remote_allowed
    int32_t err;
    uint32_t reserved;
    double pan_deg, tilt_deg;
    double pan_min, pan_max, tilt_min, tilt_max;
    double reserved2[3];
};
#pragma pack(pop)
static_assert(sizeof(GcCommandPacket) == 128,
              "GcCommandPacket layout must match the 128-byte GCC1 wire format");
static_assert(sizeof(GcStatusPacket) == 96,
              "GcStatusPacket layout must match the 96-byte GCS1 wire format");

constexpr int kRecvTimeoutMs = 250;
constexpr int kTries = 4;
constexpr auto kPollPeriod = std::chrono::milliseconds(500);

// Serializes requests: one command in flight at a time (the poller and the
// GUI share the socket).
std::mutex g_req_mutex;
int g_sock_fd = -1;
std::atomic<uint32_t> g_seq{1};

std::mutex g_status_mutex;
GalvoStatus g_last_status;
bool g_have_status = false;
std::chrono::steady_clock::time_point g_last_status_time;

std::thread g_poll_thread;
std::atomic<bool> g_poll_running{false};

void decode_status(const GcStatusPacket &pkt, GalvoStatus *out) {
    out->ok = (pkt.flags & 0x01) != 0;
    out->in_position = (pkt.flags & 0x02) != 0;
    out->calib_mode = (pkt.flags & 0x04) != 0;
    out->motors_enabled = (pkt.flags & 0x08) != 0;
    out->remote_allowed = (pkt.flags & 0x10) != 0;
    out->err = pkt.err;
    out->pan_deg = pkt.pan_deg;
    out->tilt_deg = pkt.tilt_deg;
    out->pan_min = pkt.pan_min;
    out->pan_max = pkt.pan_max;
    out->tilt_min = pkt.tilt_min;
    out->tilt_max = pkt.tilt_max;
}

// Sends one command and waits for its seq-matched reply, retrying on
// timeout. Returns true on a validated reply with err == 0.
bool request(uint16_t cmd, const double *args, int num_args,
             GalvoStatus *out) {
    std::lock_guard<std::mutex> lock(g_req_mutex);
    if (g_sock_fd < 0) {
        return false;
    }

    GcCommandPacket req;
    std::memset(&req, 0, sizeof(req));
    std::memcpy(req.magic, "GCC1", 4);
    req.version = 1;
    req.cmd = cmd;
    req.seq = g_seq.fetch_add(1);
    for (int i = 0; i < num_args && i < 14; i++) {
        req.args[i] = args[i];
    }

    for (int attempt = 0; attempt < kTries; attempt++) {
        if (send(g_sock_fd, &req, sizeof(req), 0) != sizeof(req)) {
            continue;
        }

        // Drain replies until the matching seq or the socket times out;
        // stale replies from a retried earlier request are skipped.
        GcStatusPacket rep;
        ssize_t n;
        while ((n = recv(g_sock_fd, &rep, sizeof(rep), 0)) > 0) {
            if (n != sizeof(rep) || std::memcmp(rep.magic, "GCS1", 4) != 0 ||
                rep.version != 1 || rep.seq != req.seq) {
                continue;
            }
            GalvoStatus status;
            decode_status(rep, &status);
            {
                std::lock_guard<std::mutex> slock(g_status_mutex);
                g_last_status = status;
                g_have_status = true;
                g_last_status_time = std::chrono::steady_clock::now();
            }
            if (out != nullptr) {
                *out = status;
            }
            return status.ok && status.err == 0;
        }
    }
    return false;
}

void poll_loop() {
    while (g_poll_running.load()) {
        request(GC_CMD_PING, nullptr, 0, nullptr);
        auto deadline = std::chrono::steady_clock::now() + kPollPeriod;
        while (g_poll_running.load() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

} // namespace

bool galvo_link_start(const std::string &target_ip, int target_port) {
    galvo_link_stop();

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::cerr << "galvo_link: failed to create UDP socket" << std::endl;
        return false;
    }

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(static_cast<uint16_t>(target_port));
    if (inet_pton(AF_INET, target_ip.c_str(), &dst.sin_addr) != 1) {
        std::cerr << "galvo_link: invalid target IP '" << target_ip << "'"
                  << std::endl;
        close(fd);
        return false;
    }
    // connect() filters incoming datagrams to the peer and lets us use
    // send/recv instead of sendto/recvfrom.
    if (connect(fd, reinterpret_cast<sockaddr *>(&dst), sizeof(dst)) != 0) {
        std::cerr << "galvo_link: connect failed" << std::endl;
        close(fd);
        return false;
    }
    timeval to{0, kRecvTimeoutMs * 1000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof(to));

    {
        std::lock_guard<std::mutex> lock(g_req_mutex);
        g_sock_fd = fd;
    }
    {
        std::lock_guard<std::mutex> slock(g_status_mutex);
        g_have_status = false;
    }

    g_poll_running.store(true);
    g_poll_thread = std::thread(poll_loop);
    return true;
}

void galvo_link_stop() {
    if (g_poll_running.exchange(false)) {
        if (g_poll_thread.joinable()) {
            g_poll_thread.join();
        }
    }
    std::lock_guard<std::mutex> lock(g_req_mutex);
    if (g_sock_fd >= 0) {
        close(g_sock_fd);
        g_sock_fd = -1;
    }
}

bool galvo_link_running() {
    std::lock_guard<std::mutex> lock(g_req_mutex);
    return g_sock_fd >= 0;
}

bool galvo_link_last_status(GalvoStatus *out, double *age_s) {
    std::lock_guard<std::mutex> slock(g_status_mutex);
    if (!g_have_status) {
        return false;
    }
    if (out != nullptr) {
        *out = g_last_status;
    }
    if (age_s != nullptr) {
        *age_s = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - g_last_status_time)
                     .count();
    }
    return true;
}

bool galvo_link_ping(GalvoStatus *out) {
    return request(GC_CMD_PING, nullptr, 0, out);
}

bool galvo_link_set_angles(double pan_deg, double tilt_deg, GalvoStatus *out) {
    double args[2] = {pan_deg, tilt_deg};
    return request(GC_CMD_SET_ANGLES, args, 2, out);
}

bool galvo_link_calib_mode(bool on, GalvoStatus *out) {
    double args[1] = {on ? 1.0 : 0.0};
    return request(GC_CMD_CALIB_MODE, args, 1, out);
}

bool galvo_link_set_calib(const GalvoCalibModel &model, GalvoStatus *out) {
    double args[12] = {model.base[0],   model.base[1],  model.base[2],
                       model.rot[0],    model.rot[1],   model.rot[2],
                       model.pan_sign,  model.pan_scale, model.pan_offset,
                       model.tilt_sign, model.tilt_scale, model.tilt_offset};
    return request(GC_CMD_SET_CALIB, args, 12, out);
}

bool galvo_link_save_config(GalvoStatus *out) {
    return request(GC_CMD_SAVE_CONFIG, nullptr, 0, out);
}

bool galvo_link_stop_motion(GalvoStatus *out) {
    return request(GC_CMD_STOP, nullptr, 0, out);
}
