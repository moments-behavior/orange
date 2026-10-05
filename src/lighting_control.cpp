#include "lighting_control.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
static void close_socket(socket_t s) { closesocket(s); }
static bool connect_in_progress() { return WSAGetLastError() == WSAEWOULDBLOCK; }
static bool set_nonblocking(socket_t s, bool on) {
    u_long mode = on ? 1 : 0;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
}
static void set_io_timeout(socket_t s, int ms) {
    DWORD t = (DWORD)ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&t, sizeof(t));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&t, sizeof(t));
}
constexpr int kSendFlags = 0;
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
static void close_socket(socket_t s) { close(s); }
static bool connect_in_progress() { return errno == EINPROGRESS; }
static bool set_nonblocking(socket_t s, bool on) {
    int flags = fcntl(s, F_GETFL, 0);
    if (flags < 0)
        return false;
    return fcntl(s, F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK)) == 0;
}
static void set_io_timeout(socket_t s, int ms) {
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
// A controller that drops the connection mid-send must not SIGPIPE orange.
constexpr int kSendFlags = MSG_NOSIGNAL;
#endif

std::atomic<bool> g_lighting_running{false};
std::atomic<bool> g_lighting_should_stop{false};
std::atomic<float> g_lighting_time_frac{0.0f};
std::atomic<float> g_lighting_cct_k{0.0f};
std::atomic<float> g_lighting_brightness{0.0f};
std::atomic<bool> g_lighting_zone_b{false};
std::atomic<bool> g_lighting_link_ok{false};

namespace {

// Resend the current state even when nothing changed, so a controller that
// rebooted (and came back up dark, in its own default segments) recovers.
constexpr double kKeepaliveSec = 5.0;
// Bounds every connect/send/recv, so an unreachable controller can't stall
// the thread -- or stop_lighting_thread's join -- for the OS's ~20s default.
constexpr int kHttpTimeoutMs = 1500;
constexpr int kZoneFadeTenths = 10; // WLED "tt" units (100ms): 1s fade out/in
constexpr int kRelaySettleMs = 200;

// Fraction of the local calendar day elapsed right now (0 at midnight, just
// under 1 the instant before the next midnight), with sub-second precision
// so the cycle position moves smoothly instead of jumping once per second.
float local_time_of_day_fraction() {
    using namespace std::chrono;
    system_clock::time_point now = system_clock::now();
    std::time_t now_c = system_clock::to_time_t(now);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_c);
#else
    localtime_r(&now_c, &local_tm);
#endif
    double seconds_since_midnight =
        local_tm.tm_hour * 3600.0 + local_tm.tm_min * 60.0 + local_tm.tm_sec;
    double total_seconds = duration<double>(now.time_since_epoch()).count();
    seconds_since_midnight += total_seconds - std::floor(total_seconds);
    return (float)(seconds_since_midnight / 86400.0);
}

bool send_all(socket_t s, const std::string &data) {
    size_t sent = 0;
    while (sent < data.size()) {
        int n = send(s, data.data() + sent, (int)(data.size() - sent), kSendFlags);
        if (n <= 0)
            return false;
        sent += (size_t)n;
    }
    return true;
}

std::string ipv4_to_string(uint32_t addr_host) {
    in_addr a{};
    a.s_addr = htonl(addr_host);
    char buf[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &a, buf, sizeof(buf));
    return buf;
}

// One HTTP/1.1 request to `dest` (port in dest), with `timeout_ms` bounding
// the non-blocking connect and each send/recv. Returns true on a 2xx status.
// When `response` is given, the whole response (headers + body, capped at
// 16 KB) is read into it; otherwise only the status line is read.
bool http_request(const sockaddr_in &dest, const char *method, const char *path,
                  const std::string &body, int timeout_ms, std::string *response) {
    socket_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == kInvalidSocket)
        return false;
    bool ok = false;
    do {
        if (!set_nonblocking(s, true))
            break;
        if (connect(s, (const sockaddr *)&dest, sizeof(dest)) != 0) {
            if (!connect_in_progress())
                break;
            fd_set wfds, efds;
            FD_ZERO(&wfds);
            FD_ZERO(&efds);
            FD_SET(s, &wfds);
            FD_SET(s, &efds);
            timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
            if (select((int)s + 1, nullptr, &wfds, &efds, &tv) <= 0)
                break;
            int err = 0;
            socklen_t len = sizeof(err);
            if (getsockopt(s, SOL_SOCKET, SO_ERROR, (char *)&err, &len) != 0 || err != 0)
                break;
        }
        if (!set_nonblocking(s, false))
            break;
        set_io_timeout(s, timeout_ms);

        std::string request = std::string(method) + " " + path + " HTTP/1.1\r\nHost: " +
                              ipv4_to_string(ntohl(dest.sin_addr.s_addr)) +
                              "\r\nConnection: close\r\n";
        if (!body.empty())
            request += "Content-Type: application/json\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n";
        request += "\r\n" + body;
        if (!send_all(s, request))
            break;

        std::string got;
        const size_t want = response ? 16384 : 12; // 12 = "HTTP/1.1 200"
        char chunk[2048];
        while (got.size() < want) {
            int n = recv(s, chunk, (int)std::min(sizeof(chunk), want - got.size()), 0);
            if (n <= 0)
                break;
            got.append(chunk, (size_t)n);
        }
        ok = got.size() >= 12 && got.compare(0, 7, "HTTP/1.") == 0 && got[9] == '2';
        if (response)
            *response = std::move(got);
    } while (false);
    close_socket(s);
    return ok;
}

// WLED /json/state body. Segment 0 is the PWM strip: WLED re-derives the
// warm/cool split from W and a relative CCT byte, and with the controller's
// CCT blending at 0 that split is linear -- the same crossfade
// day_night_levels() computed -- so sending W = WW+CW and CCT = CW/W
// reproduces our two white levels. Segment 1 is the relay's On/Off output
// (lit = relay energized = zone B); relay < 0 leaves it out so a fade can't
// flip it. Both segments' bounds are restated every time so the layout
// survives a controller reboot.
std::string state_json(const RgbwwLevels &lv, int relay, int master_bri,
                       int transition_tenths) {
    int w = std::min(255, lv.ww + lv.cw);
    int cct = w > 0 ? (int)std::lround(lv.cw * 255.0 / w) : 127;
    char seg0[160];
    std::snprintf(seg0, sizeof(seg0),
                  "{\"id\":0,\"start\":0,\"stop\":1,\"on\":true,\"bri\":255,\"fx\":0,"
                  "\"col\":[[%d,%d,%d,%d]],\"cct\":%d}",
                  lv.r, lv.g, lv.b, w, cct);
    std::string out = "{\"on\":true,\"bri\":" + std::to_string(master_bri) +
                      ",\"tt\":" + std::to_string(transition_tenths) +
                      ",\"seg\":[" + seg0;
    if (relay >= 0)
        out += std::string(",{\"id\":1,\"start\":1,\"stop\":2,\"on\":") +
               (relay ? "true" : "false") +
               ",\"bri\":255,\"fx\":0,\"col\":[[255,255,255,255]]}";
    out += "]}";
    return out;
}

// Sleeps in short slices so Stop Cycle stays responsive during the multi-
// second zone-switch fades. Returns false if a stop was requested.
bool sleep_unless_stopped(int ms) {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (g_lighting_should_stop.load(std::memory_order_relaxed))
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return !g_lighting_should_stop.load(std::memory_order_relaxed);
}

void lighting_thread_func(LightingConfig config) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        g_lighting_running.store(false, std::memory_order_relaxed);
        return;
    }
#endif

    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(80);
    if (inet_pton(AF_INET, config.controller_ip.c_str(), &dest.sin_addr) != 1) {
        g_lighting_link_ok.store(false, std::memory_order_relaxed);
        g_lighting_running.store(false, std::memory_order_relaxed);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    auto send = [&](const std::string &body) {
        bool ok = http_request(dest, "POST", "/json/state", body, kHttpTimeoutMs, nullptr);
        g_lighting_link_ok.store(ok, std::memory_order_relaxed);
        return ok;
    };

    const RgbwwLevels kDark{0, 0, 0, 0, 0};
    const int master_bri =
        std::clamp((int)std::lround(config.max_output * 255.0f), 0, 255);
    const int refresh_ms = (int)std::lround(
        std::clamp(config.update_interval_s, kMinUpdateIntervalS, kMaxUpdateIntervalS) *
        1000.0f);
    auto start = std::chrono::steady_clock::now();
    bool zone_known = false; // relay state is unknown until we've set it
    bool zone_b = false;
    std::string last_sent;
    auto last_send_time = std::chrono::steady_clock::time_point{};

    while (!g_lighting_should_stop.load(std::memory_order_relaxed)) {
        float t;
        if (config.sync_to_real_time) {
            t = local_time_of_day_fraction();
        } else {
            double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
            float cycle = std::max(config.cycle_seconds, 1.0f);
            t = (float)std::fmod(elapsed, (double)cycle) / cycle;
        }

        float cct_k = 0.0f, brightness = 0.0f;
        RgbwwLevels levels = day_night_levels(t, config, &cct_k, &brightness);
        bool want_b = zone_b_active(t, config);

        g_lighting_time_frac.store(t, std::memory_order_relaxed);
        g_lighting_cct_k.store(cct_k, std::memory_order_relaxed);
        g_lighting_brightness.store(brightness, std::memory_order_relaxed);

        if (!zone_known || want_b != zone_b) {
            // Only flip the relay while the strip is dark: fade out, switch,
            // fade back in. Avoids switching the relay's DC contacts under
            // load and a visible snap when a zone powers up.
            if (!send(state_json(kDark, -1, master_bri, kZoneFadeTenths))) {
                sleep_unless_stopped(refresh_ms);
                continue;
            }
            if (!sleep_unless_stopped(kZoneFadeTenths * 100 + 100))
                break;
            if (!send(state_json(kDark, want_b, master_bri, 0))) {
                sleep_unless_stopped(refresh_ms);
                continue;
            }
            zone_known = true;
            zone_b = want_b;
            g_lighting_zone_b.store(zone_b, std::memory_order_relaxed);
            if (!sleep_unless_stopped(kRelaySettleMs))
                break;
            if (send(state_json(levels, zone_b, master_bri, kZoneFadeTenths))) {
                last_sent = state_json(levels, zone_b, master_bri, 0);
                last_send_time = std::chrono::steady_clock::now();
            }
            if (!sleep_unless_stopped(kZoneFadeTenths * 100 + 100))
                break;
            continue;
        }

        std::string body = state_json(levels, zone_b, master_bri, 0);
        auto now = std::chrono::steady_clock::now();
        if (body != last_sent ||
            std::chrono::duration<double>(now - last_send_time).count() >= kKeepaliveSec) {
            if (send(body)) {
                last_sent = body;
                last_send_time = now;
            }
        }
        sleep_unless_stopped(refresh_ms);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    g_lighting_running.store(false, std::memory_order_relaxed);
}

// ---- Controller discovery ------------------------------------------------

constexpr int kWledNodePort = 65506; // WLED's "node info" broadcast port
constexpr int kScanTimeoutMs = 600;  // per-address connect/read budget on a LAN
constexpr int kScanWorkers = 64;

std::mutex g_discovery_mutex;
std::vector<DiscoveredController> g_discovered;
std::string g_scan_status;
std::atomic<bool> g_listener_active{false};
std::atomic<bool> g_scan_active{false};
std::atomic<bool> g_discovery_stop{false}; // set once, at program exit
std::thread g_listener_thread;
std::thread g_scan_thread;

void record_controller(DiscoveredController c) {
    std::lock_guard<std::mutex> lock(g_discovery_mutex);
    for (auto &e : g_discovered) {
        if (e.ip == c.ip) {
            if (!c.name.empty())
                e.name = c.name;
            if (!c.mac.empty())
                e.mac = c.mac;
            return;
        }
    }
    g_discovered.push_back(std::move(c));
}

// First `"key":"value"` string in `json`. Good enough for WLED's /json/info,
// where "name" and "mac" are unique top-level string fields.
std::string json_string_field(const std::string &json, const char *key) {
    std::string needle = std::string("\"") + key + "\":\"";
    size_t p = json.find(needle);
    if (p == std::string::npos)
        return "";
    p += needle.size();
    size_t e = json.find('"', p);
    return e == std::string::npos ? "" : json.substr(p, e - p);
}

std::string format_mac(const std::string &hex) {
    if (hex.size() != 12)
        return hex;
    std::string out;
    for (size_t i = 0; i < 12; i += 2) {
        if (i)
            out += ':';
        out += hex.substr(i, 2);
    }
    return out;
}

// Asks `addr` (host byte order) for WLED's /json/info and records it if it
// answers like a WLED controller (not just any web server on port 80).
bool probe_wled(uint32_t addr, int timeout_ms) {
    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(80);
    dest.sin_addr.s_addr = htonl(addr);
    std::string resp;
    if (!http_request(dest, "GET", "/json/info", "", timeout_ms, &resp))
        return false;
    if (resp.find("\"leds\":{") == std::string::npos ||
        resp.find("\"ver\":\"") == std::string::npos)
        return false;
    record_controller({ipv4_to_string(addr), json_string_field(resp, "name"),
                       format_mac(json_string_field(resp, "mac"))});
    return true;
}

struct LocalSubnet {
    uint32_t addr, mask; // host byte order
};

std::vector<LocalSubnet> local_ipv4_subnets() {
    std::vector<LocalSubnet> out;
#ifdef _WIN32
    ULONG size = 16384;
    std::vector<unsigned char> buf(size);
    const ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                        GAA_FLAG_SKIP_DNS_SERVER;
    ULONG rc = GetAdaptersAddresses(AF_INET, flags, nullptr,
                                    (IP_ADAPTER_ADDRESSES *)buf.data(), &size);
    if (rc == ERROR_BUFFER_OVERFLOW) {
        buf.resize(size);
        rc = GetAdaptersAddresses(AF_INET, flags, nullptr,
                                  (IP_ADAPTER_ADDRESSES *)buf.data(), &size);
    }
    if (rc != NO_ERROR)
        return out;
    for (auto *a = (IP_ADAPTER_ADDRESSES *)buf.data(); a; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
            continue;
        for (auto *u = a->FirstUnicastAddress; u; u = u->Next) {
            if (u->Address.lpSockaddr->sa_family != AF_INET)
                continue;
            uint32_t addr =
                ntohl(((sockaddr_in *)u->Address.lpSockaddr)->sin_addr.s_addr);
            int prefix = u->OnLinkPrefixLength;
            uint32_t mask = prefix <= 0 ? 0 : 0xFFFFFFFFu << (32 - std::min(prefix, 32));
            out.push_back({addr, mask});
        }
    }
#else
    ifaddrs *ifs = nullptr;
    if (getifaddrs(&ifs) != 0)
        return out;
    for (ifaddrs *i = ifs; i; i = i->ifa_next) {
        if (!i->ifa_addr || !i->ifa_netmask || i->ifa_addr->sa_family != AF_INET)
            continue;
        if (!(i->ifa_flags & IFF_UP) || (i->ifa_flags & IFF_LOOPBACK))
            continue;
        out.push_back({ntohl(((sockaddr_in *)i->ifa_addr)->sin_addr.s_addr),
                       ntohl(((sockaddr_in *)i->ifa_netmask)->sin_addr.s_addr)});
    }
    freeifaddrs(ifs);
#endif
    return out;
}

void scan_thread_func() {
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif
    std::vector<uint32_t> targets;
    std::string ranges;
    for (LocalSubnet sn : local_ipv4_subnets()) {
        if ((sn.addr & 0xFFFF0000u) == 0xA9FE0000u)
            continue; // 169.254/16 link-local: 65k addresses, no DHCP anyway
        uint32_t mask = sn.mask;
        if (mask < 0xFFFFFC00u)
            mask = 0xFFFFFF00u; // wider than /22: just this machine's /24
        if (mask >= 0xFFFFFFFEu)
            continue; // /31-/32: nothing else on the link
        uint32_t net = sn.addr & mask, bcast = net | ~mask;
        for (uint32_t a = net + 1; a < bcast; a++)
            if (a != sn.addr)
                targets.push_back(a);
        int prefix = 0;
        for (uint32_t m = mask; m; m <<= 1)
            prefix++;
        if (!ranges.empty())
            ranges += ", ";
        ranges += ipv4_to_string(net) + "/" + std::to_string(prefix);
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

    {
        std::lock_guard<std::mutex> lock(g_discovery_mutex);
        g_scan_status = targets.empty() ? "no local network to scan"
                                        : "scanning " + ranges + "...";
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    int n_workers = (int)std::min<size_t>(kScanWorkers, targets.size());
    for (int w = 0; w < n_workers; w++)
        workers.emplace_back([&] {
            while (!g_discovery_stop.load(std::memory_order_relaxed)) {
                size_t i = next.fetch_add(1);
                if (i >= targets.size())
                    break;
                probe_wled(targets[i], kScanTimeoutMs);
            }
        });
    for (auto &w : workers)
        w.join();
    if (!targets.empty()) {
        std::lock_guard<std::mutex> lock(g_discovery_mutex);
        g_scan_status = "scanned " + ranges;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    g_scan_active.store(false);
}

void listener_thread_func() {
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif
    socket_t s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s != kInvalidSocket) {
        int yes = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
        sockaddr_in local{};
        local.sin_family = AF_INET;
        local.sin_port = htons(kWledNodePort);
        local.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(s, (const sockaddr *)&local, sizeof(local)) == 0) {
            set_io_timeout(s, 500); // so the exit-time stop is noticed promptly
            while (!g_discovery_stop.load(std::memory_order_relaxed)) {
                // WLED node info packet: [255, 1, ip(4), name(32), ...].
                unsigned char pkt[64];
                sockaddr_in from{};
                socklen_t from_len = sizeof(from);
                int n = recvfrom(s, (char *)pkt, sizeof(pkt), 0, (sockaddr *)&from,
                                 &from_len);
                if (n < 38 || pkt[0] != 255 || pkt[1] != 1)
                    continue;
                uint32_t addr = ntohl(from.sin_addr.s_addr);
                char name[33];
                std::memcpy(name, pkt + 6, 32);
                name[32] = 0;
                if (!probe_wled(addr, kHttpTimeoutMs)) // also fetches the MAC
                    record_controller({ipv4_to_string(addr), name, ""});
                break; // one controller is all that's needed
            }
        }
        close_socket(s);
    }
#ifdef _WIN32
    WSACleanup();
#endif
    g_listener_active.store(false);
}

// Joins any still-running discovery thread at program exit (a joinable
// std::thread destroyed at static teardown would std::terminate). Declared
// after the state above so it's destroyed -- and joins -- before that state.
struct DiscoveryShutdown {
    ~DiscoveryShutdown() {
        g_discovery_stop.store(true);
        if (g_listener_thread.joinable())
            g_listener_thread.join();
        if (g_scan_thread.joinable())
            g_scan_thread.join();
    }
} g_discovery_shutdown;

} // namespace

void start_controller_listener() {
    if (g_listener_active.load())
        return;
    if (g_listener_thread.joinable())
        g_listener_thread.join();
    g_listener_active.store(true);
    g_listener_thread = std::thread(listener_thread_func);
}

void start_controller_scan() {
    if (g_scan_active.load())
        return;
    if (g_scan_thread.joinable())
        g_scan_thread.join();
    g_scan_active.store(true);
    g_scan_thread = std::thread(scan_thread_func);
}

bool controller_listener_active() { return g_listener_active.load(); }
bool controller_scan_active() { return g_scan_active.load(); }

std::string controller_scan_status() {
    std::lock_guard<std::mutex> lock(g_discovery_mutex);
    return g_scan_status;
}

std::vector<DiscoveredController> discovered_controllers() {
    std::lock_guard<std::mutex> lock(g_discovery_mutex);
    return g_discovered;
}

float evaluate_spline(const EditableSpline &spline, float x) {
    const auto &pts = spline.points;
    if (pts.empty())
        return 0.0f;
    if (pts.size() == 1)
        return std::clamp(pts[0].y, 0.0f, 1.0f);

    x = std::clamp(x, 0.0f, 1.0f);
    // Find the segment [pts[i], pts[i+1]] containing x. Assumes points are
    // sorted by x -- the editor in lighting_panel.cpp enforces that.
    size_t i = 0;
    while (i + 2 < pts.size() && pts[i + 1].x < x)
        i++;

    const SplinePoint &p1 = pts[i];
    const SplinePoint &p2 = pts[i + 1];
    // The cycle is periodic: x=1 is the same instant as x=0 (and the editor
    // keeps those two endpoints at the same level), so the neighbor before
    // the first point is the second-to-last point, and the one after the
    // last point is the second point. That gives the same slope on both
    // sides of the wrap, so the repeat is smooth, not just continuous.
    const size_t n = pts.size();
    const SplinePoint &p0 = (i == 0) ? pts[n - 2] : pts[i - 1];
    const SplinePoint &p3 = (i + 2 < n) ? pts[i + 2] : pts[1];

    float span = p2.x - p1.x;
    float t = span > 1e-6f ? (x - p1.x) / span : 0.0f;
    float t2 = t * t, t3 = t2 * t;

    // Standard Catmull-Rom basis, uniform parameterization (t in [0,1] per
    // segment regardless of the segment's actual x-width). Good enough for a
    // handful of hand-placed points; exact x-spacing fidelity isn't the goal.
    float y = 0.5f * ((2.0f * p1.y) + (-p0.y + p2.y) * t +
                      (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                      (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3);
    return std::clamp(y, 0.0f, 1.0f);
}

EditableSpline make_default_bump_spline() {
    return EditableSpline{{{0.0f, 0.0f},
                          {0.25f, 0.0f},
                          {0.5f, 1.0f},
                          {0.75f, 0.0f},
                          {1.0f, 0.0f}}};
}

EditableSpline make_default_flat_spline() {
    return EditableSpline{{{0.0f, 0.0f}, {1.0f, 0.0f}}};
}

RgbwwLevels day_night_levels(float time_frac, const LightingConfig &config,
                             float *out_cct_k, float *out_brightness) {
    float brightness = evaluate_spline(config.brightness_curve, time_frac);
    float cct_norm = evaluate_spline(config.cct_curve, time_frac);
    float cct_k = config.cct_min_k + (config.cct_max_k - config.cct_min_k) * cct_norm;

    if (out_cct_k)
        *out_cct_k = cct_k;
    if (out_brightness)
        *out_brightness = brightness;

    // Simple linear WW/CW crossfade to hit the target color temperature.
    // Not a physically exact blackbody blend, but it's the same approach
    // used by essentially every consumer CCT LED controller.
    float cct_span = std::max(config.cct_max_k - config.cct_min_k, 1.0f);
    float cool_frac = std::clamp((cct_k - config.cct_min_k) / cct_span, 0.0f, 1.0f);

    RgbwwLevels levels{};
    levels.r = (uint8_t)std::lround(
        evaluate_spline(config.r_curve, time_frac) * 255.0f);
    levels.g = (uint8_t)std::lround(
        evaluate_spline(config.g_curve, time_frac) * 255.0f);
    levels.b = (uint8_t)std::lround(
        evaluate_spline(config.b_curve, time_frac) * 255.0f);
    levels.ww = (uint8_t)std::lround(brightness * (1.0f - cool_frac) * 255.0f);
    levels.cw = (uint8_t)std::lround(brightness * cool_frac * 255.0f);
    return levels;
}

bool zone_b_active(float time_frac, const LightingConfig &config) {
    switch (config.zone_mode) {
    case ZoneMode::AlwaysA:
        return false;
    case ZoneMode::AlwaysB:
        return true;
    case ZoneMode::Scheduled:
        break;
    }
    float s = config.zone_b_start, e = config.zone_b_end;
    if (s <= e)
        return time_frac >= s && time_frac < e;
    return time_frac >= s || time_frac < e; // window wraps past midnight
}

void start_lighting_thread(std::thread &thread_out,
                           const LightingConfig &config) {
    g_lighting_should_stop.store(false, std::memory_order_relaxed);
    g_lighting_link_ok.store(false, std::memory_order_relaxed);
    g_lighting_running.store(true, std::memory_order_relaxed);
    thread_out = std::thread(lighting_thread_func, config);
}

void stop_lighting_thread(std::thread &thread_out) {
    g_lighting_should_stop.store(true, std::memory_order_relaxed);
    if (thread_out.joinable())
        thread_out.join();
}

namespace {

std::string spline_to_field(const EditableSpline &spline) {
    std::ostringstream oss;
    for (size_t i = 0; i < spline.points.size(); i++) {
        if (i)
            oss << ';';
        oss << spline.points[i].x << ',' << spline.points[i].y;
    }
    return oss.str();
}

// Throws std::invalid_argument via std::stof on a malformed field -- caught
// by load_lighting_config's caller-facing try/catch, since this only ever
// runs on a file that could have been hand-edited or from an older format.
EditableSpline spline_from_field(const std::string &field) {
    EditableSpline spline;
    std::istringstream iss(field);
    std::string point_str;
    while (std::getline(iss, point_str, ';')) {
        size_t comma = point_str.find(',');
        if (comma == std::string::npos)
            continue;
        float x = std::stof(point_str.substr(0, comma));
        float y = std::stof(point_str.substr(comma + 1));
        spline.points.push_back({x, y});
    }
    // Curves wrap seamlessly: the end of the cycle always matches its start
    // (presets saved before that rule may not have).
    if (spline.points.size() >= 2)
        spline.points.back().y = spline.points.front().y;
    return spline;
}

} // namespace

bool save_lighting_config(const LightingConfig &config, const std::string &path) {
    std::ofstream ofs(path);
    if (!ofs)
        return false;
    ofs << "controller_ip=" << config.controller_ip << '\n';
    ofs << "cct_min_k=" << config.cct_min_k << '\n';
    ofs << "cct_max_k=" << config.cct_max_k << '\n';
    ofs << "cycle_seconds=" << config.cycle_seconds << '\n';
    ofs << "sync_to_real_time=" << (config.sync_to_real_time ? 1 : 0) << '\n';
    ofs << "max_output=" << config.max_output << '\n';
    ofs << "update_interval_s=" << config.update_interval_s << '\n';
    ofs << "zone_mode=" << (int)config.zone_mode << '\n';
    ofs << "zone_b_start=" << config.zone_b_start << '\n';
    ofs << "zone_b_end=" << config.zone_b_end << '\n';
    ofs << "cct_curve=" << spline_to_field(config.cct_curve) << '\n';
    ofs << "brightness_curve=" << spline_to_field(config.brightness_curve) << '\n';
    ofs << "r_curve=" << spline_to_field(config.r_curve) << '\n';
    ofs << "g_curve=" << spline_to_field(config.g_curve) << '\n';
    ofs << "b_curve=" << spline_to_field(config.b_curve) << '\n';
    return (bool)ofs;
}

bool load_lighting_config(LightingConfig &out, const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs)
        return false;
    try {
        LightingConfig loaded; // parse into a scratch copy so a partially-
                               // malformed file can't half-overwrite `out`
        std::string line;
        while (std::getline(ifs, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            std::string key = line.substr(0, eq);
            std::string value = line.substr(eq + 1);
            if (key == "controller_ip")
                loaded.controller_ip = value;
            else if (key == "cct_min_k")
                loaded.cct_min_k = std::stof(value);
            else if (key == "cct_max_k")
                loaded.cct_max_k = std::stof(value);
            else if (key == "cycle_seconds")
                loaded.cycle_seconds = std::stof(value);
            else if (key == "sync_to_real_time")
                loaded.sync_to_real_time = (value != "0");
            else if (key == "max_output")
                loaded.max_output = std::clamp(std::stof(value), 0.0f, 1.0f);
            else if (key == "update_interval_s")
                loaded.update_interval_s = std::clamp(
                    std::stof(value), kMinUpdateIntervalS, kMaxUpdateIntervalS);
            else if (key == "zone_mode")
                loaded.zone_mode = (ZoneMode)std::clamp(std::stoi(value), 0, 2);
            else if (key == "zone_b_start")
                loaded.zone_b_start = std::clamp(std::stof(value), 0.0f, 1.0f);
            else if (key == "zone_b_end")
                loaded.zone_b_end = std::clamp(std::stof(value), 0.0f, 1.0f);
            else if (key == "cct_curve")
                loaded.cct_curve = spline_from_field(value);
            else if (key == "brightness_curve")
                loaded.brightness_curve = spline_from_field(value);
            else if (key == "r_curve")
                loaded.r_curve = spline_from_field(value);
            else if (key == "g_curve")
                loaded.g_curve = spline_from_field(value);
            else if (key == "b_curve")
                loaded.b_curve = spline_from_field(value);
        }
        out = loaded;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}
