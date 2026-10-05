#ifndef ORANGE_LIGHTING_CONTROL
#define ORANGE_LIGHTING_CONTROL

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

// A single user-draggable control point on a lighting curve. x is a time-of-
// day fraction [0,1]; y is a normalized channel level [0,1].
struct SplinePoint {
    float x, y;
};

// A small, user-editable curve over the day/night cycle, Catmull-Rom
// interpolated between points. Points must stay sorted by x, and the editor
// (lighting_panel.cpp) pins point 0 to x=0 and the last point to x=1 so the
// curve always spans the full cycle. Those two endpoints always share the
// same y, and the curve treats the cycle as periodic (points just past
// midnight are neighbors of points just before it), so it repeats with no
// jump and no kink.
struct EditableSpline {
    std::vector<SplinePoint> points;
};

// Evaluates `spline` at `x` (clamped to [0,1]), Catmull-Rom interpolated
// between the surrounding control points and clamped to [0,1] output --
// LED levels can't go negative or overshoot, so overshoot past the end
// points is simply clipped rather than reproduced.
float evaluate_spline(const EditableSpline &spline, float x);

// Reproduces the day/night cycle's original fixed half-sine shape: 0 until
// 0.25, up to 1 at 0.5 (solar noon), back to 0 by 0.75, 0 for the rest of
// the cycle. Used as the default for the CCT and brightness curves.
EditableSpline make_default_bump_spline();

// Flat 0 for the whole cycle. Used as the default for the R/G/B curves --
// they're always active, but start out contributing nothing until shaped.
EditableSpline make_default_flat_spline();

// Which of the two strips is powered. Both zones share the controller's PWM
// channels; a single SPDT relay on the controller's GPIO18 routes the strip
// supply's V+ to one zone or the other. Zone A is wired to the relay's NC
// contact (powered whenever the relay is de-energized, including during
// boot), zone B to NO.
enum class ZoneMode : int { AlwaysA = 0, AlwaysB = 1, Scheduled = 2 };

// Update interval bounds. The controller answers a request in ~20 ms, and
// updates go out one at a time, so 10 Hz leaves plenty of headroom; 5 min is
// as coarse as a day-long cycle sensibly gets.
constexpr float kMinUpdateIntervalS = 0.1f;
constexpr float kMaxUpdateIntervalS = 300.0f;

// Drives a WLED controller (ESP32, wired Ethernet) over its JSON HTTP API to
// simulate a day/night lighting cycle. The controller must be configured
// with LED 0 = a 5-channel "PWM RGBCCT" output and LED 1 = an "On/Off"
// output on GPIO18 driving the zone relay; the app writes WLED segment 0
// (strip) and segment 1 (relay) on every update.
struct LightingConfig {
    std::string controller_ip = "10.101.30.40";
    float cct_min_k = 2700.0f; // W1: the strip's warm-white channel
    float cct_max_k = 6500.0f; // W2: the strip's cool-white channel
    float cycle_seconds = 86400.0f; // real seconds per simulated 24h day (default: 1 real day)

    // When true (the default), the cycle position tracks the host machine's
    // actual local time of day (t=0 at midnight) instead of cycle_seconds --
    // the lights simply follow real time. cycle_seconds is ignored (and
    // disabled in the GUI) while this is on.
    bool sync_to_real_time = true;

    // Hard cap on every channel's PWM duty (0-1), sent as WLED's master
    // brightness. The curves' 0-1 levels are fractions of this cap.
    float max_output = 0.5f;

    // Seconds between LED updates, clamped to
    // [kMinUpdateIntervalS, kMaxUpdateIntervalS].
    float update_interval_s = 1.0f;

    ZoneMode zone_mode = ZoneMode::AlwaysA;
    // Scheduled mode: zone B is powered for cycle fractions in
    // [zone_b_start, zone_b_end), zone A the rest of the time. start > end
    // wraps past midnight (e.g. 0.75 -> 0.25 = 18:00 to 06:00).
    float zone_b_start = 0.5f;
    float zone_b_end = 1.0f;

    EditableSpline cct_curve = make_default_bump_spline();        // 0 -> cct_min_k, 1 -> cct_max_k
    EditableSpline brightness_curve = make_default_bump_spline(); // 0 -> off, 1 -> full
    EditableSpline r_curve = make_default_flat_spline();
    EditableSpline g_curve = make_default_flat_spline();
    EditableSpline b_curve = make_default_flat_spline();
};

// Latest computed state, published by the lighting thread for the GUI to
// poll/plot without touching the thread's internals.
extern std::atomic<bool> g_lighting_running;
extern std::atomic<bool> g_lighting_should_stop;
extern std::atomic<float> g_lighting_time_frac; // 0-1 position in the cycle
extern std::atomic<float> g_lighting_cct_k;
extern std::atomic<float> g_lighting_brightness; // 0-1
extern std::atomic<bool> g_lighting_zone_b;      // relay energized (zone B powered)
extern std::atomic<bool> g_lighting_link_ok;     // last request to the controller succeeded

struct RgbwwLevels {
    uint8_t r, g, b, ww, cw;
};

// Pure function so the GUI can preview the curves (e.g. plot a full cycle)
// without the thread running. Levels are before the max_output cap.
RgbwwLevels day_night_levels(float time_frac, const LightingConfig &config,
                             float *out_cct_k = nullptr,
                             float *out_brightness = nullptr);

// Whether zone B (relay energized) should be powered at `time_frac`.
bool zone_b_active(float time_frac, const LightingConfig &config);

// Starts/stops the background thread that pushes the current levels and
// zone to config.controller_ip. orange only ever runs one lighting thread at
// a time (mirrors the PTP-logging worker's pattern in orange.cpp); stop is
// safe to call when not running. Stopping leaves the lights at their last
// state.
void start_lighting_thread(std::thread &thread_out, const LightingConfig &config);
void stop_lighting_thread(std::thread &thread_out);

// Finding the controller on whatever network it ends up on. Two ways, both
// running on background threads and filling a shared results list:
//  - Listening: WLED broadcasts a small "node info" UDP packet (port 65506)
//    with its IP and name. Passive -- no traffic sent -- and stops itself as
//    soon as it hears one controller.
//  - Scanning: asks every address on this machine's local subnets for
//    WLED's /json/info, which also yields the MAC. Subnets wider than /22
//    are narrowed to this machine's own /24, and link-local (169.254/16)
//    is skipped.
// Both are safe to call repeatedly (no-op while already running); any
// running discovery is stopped automatically at program exit.
struct DiscoveredController {
    std::string ip;
    std::string name;
    std::string mac; // "aa:bb:cc:dd:ee:ff"; empty if only heard, not queried
};
void start_controller_listener();
void start_controller_scan();
bool controller_listener_active();
bool controller_scan_active();
std::string controller_scan_status(); // e.g. "scanned 10.101.30.0/24"
std::vector<DiscoveredController> discovered_controllers();

// Saves/loads a LightingConfig (including all five curves) to/from a
// simple human-readable key=value text file at `path`. Backs the GUI's
// preset save/load slots. save returns false on I/O failure. load returns
// false if the file is missing or malformed, leaving `out` untouched.
// Unknown keys (e.g. from older presets) are ignored.
bool save_lighting_config(const LightingConfig &config, const std::string &path);
bool load_lighting_config(LightingConfig &out, const std::string &path);

#endif
