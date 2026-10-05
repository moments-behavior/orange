#include "lighting_panel.h"

#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

// Extra vertical room above/below the [0,1] data range so points sitting at
// y=0 or y=1 (very common -- it's where the default curve starts/ends)
// aren't rendered right on the plot border, which made them fiddly to grab.
constexpr double kYAxisPadding = 0.1;

// Standard blackbody color-temperature approximation (Tanner Helland's
// fit), valid over the range these curves actually use. Only needed for the
// background preview swatch below -- not physically exact, just close
// enough to look right.
ImVec4 kelvin_to_rgb(float kelvin) {
    float temp = kelvin / 100.0f;
    float r, g, b;

    if (temp <= 66.0f) {
        r = 255.0f;
    } else {
        r = 329.698727446f * powf(temp - 60.0f, -0.1332047592f);
    }

    if (temp <= 66.0f) {
        g = 99.4708025861f * logf(temp) - 161.1195681661f;
    } else {
        g = 288.1221695283f * powf(temp - 60.0f, -0.0755148492f);
    }

    if (temp >= 66.0f) {
        b = 255.0f;
    } else if (temp <= 19.0f) {
        b = 0.0f;
    } else {
        b = 138.5177312231f * logf(temp - 10.0f) - 305.0447927307f;
    }

    return ImVec4(std::clamp(r, 0.0f, 255.0f) / 255.0f,
                 std::clamp(g, 0.0f, 255.0f) / 255.0f,
                 std::clamp(b, 0.0f, 255.0f) / 255.0f, 1.0f);
}

// What the fixture would actually look like at `time_frac`: the CCT-blended
// white channels (scaled by brightness) plus R/G/B added on top, since on
// the real strip they're separate LEDs whose light mixes additively -- all
// scaled by the max_output cap the controller applies.
ImVec4 preview_color(const LightingConfig &config, float time_frac) {
    float cct_k = 0.0f, brightness = 0.0f;
    RgbwwLevels levels = day_night_levels(time_frac, config, &cct_k, &brightness);
    ImVec4 white = kelvin_to_rgb(cct_k);
    float cap = config.max_output;
    return ImVec4(
        cap * std::clamp(brightness * white.x + levels.r / 255.0f, 0.0f, 1.0f),
        cap * std::clamp(brightness * white.y + levels.g / 255.0f, 0.0f, 1.0f),
        cap * std::clamp(brightness * white.z + levels.b / 255.0f, 0.0f, 1.0f), 1.0f);
}

// Number of solid-color bands used to approximate the preview-color
// gradient across the cycle. Coarse enough to be cheap, fine enough that
// individual bands aren't obviously visible as steps.
constexpr int kBackgroundBands = 48;

// Paints the plot's background with `preview_color` swatches so the graph
// shows what the light will actually look like at each point in the cycle.
// Must be called after the axis limits are set up (it reads them for the
// vertical extent) and before the curve lines are plotted (draw order is
// z-order, and the lines need to render on top of this).
void draw_background_bands(const LightingConfig &config) {
    ImPlotRect limits = ImPlot::GetPlotLimits();
    ImDrawList *draw_list = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();
    for (int i = 0; i < kBackgroundBands; i++) {
        float x0 = (float)i / kBackgroundBands;
        float x1 = (float)(i + 1) / kBackgroundBands;
        ImVec4 color = preview_color(config, (x0 + x1) * 0.5f);
        ImVec2 p0 = ImPlot::PlotToPixels(ImPlotPoint(x0, limits.Y.Min));
        ImVec2 p1 = ImPlot::PlotToPixels(ImPlotPoint(x1, limits.Y.Max));
        draw_list->AddRectFilled(p0, p1, ImGui::ColorConvertFloat4ToU32(color));
    }
    ImPlot::PopPlotClipRect();
}

// Thin strip in the plot's top padding (above y=1, where no curve point can
// go) showing which zone the relay powers across the cycle, labeled A/B.
void draw_zone_strip(const LightingConfig &config) {
    constexpr int kSamples = 288; // 5-minute resolution over a day
    const ImU32 kZoneA = IM_COL32(110, 110, 125, 255);
    const ImU32 kZoneB = IM_COL32(0, 170, 160, 255);
    ImDrawList *draw_list = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();
    int run_start = 0;
    bool run_b = zone_b_active(0.5f / kSamples, config);
    for (int i = 1; i <= kSamples; i++) {
        bool b = i < kSamples && zone_b_active((i + 0.5f) / kSamples, config);
        if (i < kSamples && b == run_b)
            continue;
        ImVec2 p0 = ImPlot::PlotToPixels(
            ImPlotPoint((double)run_start / kSamples, 1.0 + kYAxisPadding * 0.9));
        ImVec2 p1 = ImPlot::PlotToPixels(
            ImPlotPoint((double)i / kSamples, 1.0 + kYAxisPadding * 0.35));
        draw_list->AddRectFilled(p0, p1, run_b ? kZoneB : kZoneA);
        const char *label = run_b ? "B" : "A";
        ImVec2 ts = ImGui::CalcTextSize(label);
        if (p1.x - p0.x > ts.x + 4.0f)
            draw_list->AddText(ImVec2((p0.x + p1.x - ts.x) * 0.5f,
                                      (p0.y + p1.y - ts.y) * 0.5f),
                               IM_COL32_WHITE, label);
        run_start = i;
        run_b = b;
    }
    ImPlot::PopPlotClipRect();
}

// "HH:MM" text field bound to a 0-1 fraction of the (possibly simulated)
// day. `buf`/`shown` are the caller's persistent edit state: buf is only
// re-formatted when `value` changes from elsewhere (e.g. a preset load), so
// an in-progress edit isn't clobbered. Edits commit when the field loses
// focus or Enter is pressed; text that doesn't parse reverts.
void time_of_day_input(const char *id, float &value, char (&buf)[8], float &shown) {
    if (value != shown) {
        int minutes = (int)std::lround(value * 1440.0f);
        std::snprintf(buf, sizeof(buf), "%02d:%02d", minutes / 60, minutes % 60);
        shown = value;
    }
    ImGui::SetNextItemWidth(55.0f);
    ImGui::InputText(id, buf, sizeof(buf));
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        int h = -1, m = -1;
        if (std::sscanf(buf, "%d:%d", &h, &m) == 2 && h >= 0 && m >= 0 && m < 60 &&
            (h < 24 || (h == 24 && m == 0)))
            value = (h * 60 + m) / 1440.0f;
        shown = -1.0f; // re-format next frame: normalizes "6:5" or reverts junk
    }
}

// Formats the Y1 axis's raw 0-1 value as a Kelvin number, used while the CCT
// curve is being edited so the axis reads in the units that curve actually
// controls. `user_data` is the LightingConfig, for cct_min_k/cct_max_k.
int format_cct_axis(double value, char *buff, int size, void *user_data) {
    const LightingConfig *cfg = static_cast<const LightingConfig *>(user_data);
    float k = cfg->cct_min_k + (float)value * (cfg->cct_max_k - cfg->cct_min_k);
    return snprintf(buff, size, "%.0fK", k);
}

// Formats the X1 axis's raw 0-1 cycle position as a 24h clock time, used
// when the cycle is synced to real time so the axis reads in actual time of
// day instead of an abstract fraction.
int format_time_of_day_axis(double value, char *buff, int size, void *) {
    double total_minutes = std::clamp(value, 0.0, 1.0) * 24.0 * 60.0;
    int hour = ((int)(total_minutes / 60.0)) % 24;
    int minute = ((int)total_minutes) % 60;
    return snprintf(buff, size, "%02d:%02d", hour, minute);
}

enum class CurveId { Brightness, Cct, R, G, B };

EditableSpline &curve_ref(LightingConfig &config, CurveId id) {
    switch (id) {
    case CurveId::Brightness:
        return config.brightness_curve;
    case CurveId::Cct:
        return config.cct_curve;
    case CurveId::R:
        return config.r_curve;
    case CurveId::G:
        return config.g_curve;
    case CurveId::B:
        return config.b_curve;
    }
    return config.brightness_curve; // unreachable
}

ImVec4 curve_color(CurveId id) {
    switch (id) {
    case CurveId::Brightness:
        return ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
    case CurveId::Cct:
        return ImVec4(1.0f, 0.65f, 0.0f, 1.0f);
    case CurveId::R:
        return ImVec4(0.9f, 0.2f, 0.2f, 1.0f);
    case CurveId::G:
        return ImVec4(0.2f, 0.85f, 0.2f, 1.0f);
    case CurveId::B:
        return ImVec4(0.3f, 0.5f, 1.0f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1); // unreachable
}

const char *curve_name(CurveId id) {
    switch (id) {
    case CurveId::Brightness:
        return "brightness";
    case CurveId::Cct:
        return "CCT";
    case CurveId::R:
        return "R";
    case CurveId::G:
        return "G";
    case CurveId::B:
        return "B";
    }
    return ""; // unreachable
}

// Renders `spline` as a line in the currently-open ImPlot plot. When
// `editable` is true its control points become drag handles: left-drag to
// reshape (x clamped between neighbors so points can't cross and points
// can't be dragged past the domain edges), double-click empty plot space to
// add a point, right-click a point to remove it. The two endpoints (x=0 and
// x=1) are pinned/undeletable so every curve always spans the full cycle.
// `id_base` must be unique per curve so drag-point IDs don't collide when
// multiple splines are shown in the same plot.
void plot_spline(const char *name, EditableSpline &spline,
                 const ImVec4 &color, bool editable, int id_base) {
    static float xs[128], ys[128];
    for (int i = 0; i < 128; i++) {
        float x = (float)i / 127.0f;
        xs[i] = x;
        ys[i] = evaluate_spline(spline, x);
    }
    ImPlot::SetNextLineStyle(color);
    ImPlot::PlotLine(name, xs, ys, 128);

    if (!editable)
        return;

    auto &pts = spline.points;
    int remove_index = -1;
    for (int i = 0; i < (int)pts.size(); i++) {
        double x = pts[i].x, y = pts[i].y;
        bool hovered = false;
        ImPlot::DragPoint(id_base + i, &x, &y, color, 5,
                          ImPlotDragToolFlags_Delayed, nullptr, &hovered,
                          nullptr);

        // Pin endpoints to the domain edges; clamp interior points between
        // their neighbors so evaluate_spline's sorted-by-x assumption holds.
        if (i == 0)
            x = 0.0;
        else if (i == (int)pts.size() - 1)
            x = 1.0;
        else
            x = std::clamp(x, (double)pts[i - 1].x + 0.001,
                           (double)pts[i + 1].x - 0.001);
        pts[i].x = (float)x;
        pts[i].y = (float)std::clamp(y, 0.0, 1.0);
        // The two endpoints share one level so the cycle wraps seamlessly:
        // dragging either one moves the other with it.
        if (i == 0)
            pts.back().y = pts[0].y;
        else if (i == (int)pts.size() - 1)
            pts.front().y = pts[i].y;

        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
            i != 0 && i != (int)pts.size() - 1 && pts.size() > 2) {
            remove_index = i;
        }
    }
    if (remove_index >= 0)
        pts.erase(pts.begin() + remove_index);

    if (ImPlot::IsPlotHovered() &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        ImPlotPoint mp = ImPlot::GetPlotMousePos();
        float nx = std::clamp((float)mp.x, 0.0f, 1.0f);
        float ny = std::clamp((float)mp.y, 0.0f, 1.0f);
        size_t insert_at = 0;
        while (insert_at < pts.size() && pts[insert_at].x < nx)
            insert_at++;
        pts.insert(pts.begin() + insert_at, SplinePoint{nx, ny});
    }
}

} // namespace

void render_lighting_panel(LightingConfig &config, std::thread &thread_ref) {
    static CurveId selected_curve = CurveId::Brightness;
    static bool listener_started = false;
    if (!listener_started) {
        start_controller_listener();
        listener_started = true;
    }

    ImGui::Begin("Lighting");

    bool running = g_lighting_running.load(std::memory_order_relaxed);
    if (running)
        ImGui::BeginDisabled();
    static char ip_buf[64] = "";
    static bool ip_buf_init = false;
    if (!ip_buf_init) {
        std::snprintf(ip_buf, sizeof(ip_buf), "%s", config.controller_ip.c_str());
        ip_buf_init = true;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Controller IP:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220.0f * 2.0f / 3.0f);
    if (ImGui::InputText("##ControllerIP", ip_buf, sizeof(ip_buf)))
        config.controller_ip = ip_buf;
    ImGui::SameLine();
    bool scanning = controller_scan_active();
    if (scanning)
        ImGui::BeginDisabled();
    if (ImGui::Button(scanning ? "Scanning...###FindController"
                               : "Find Controller###FindController"))
        start_controller_scan();
    if (scanning)
        ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Max Output:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60.0f);
    float max_pct = config.max_output * 100.0f;
    if (ImGui::InputFloat("##MaxOutput", &max_pct, 0.0f, 0.0f, "%.0f %%"))
        config.max_output = std::clamp(max_pct, 1.0f, 100.0f) / 100.0f;
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Update Every:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70.0f);
    ImGui::InputFloat("##UpdateInterval", &config.update_interval_s, 0.0f, 0.0f, "%.2f s");
    if (ImGui::IsItemDeactivatedAfterEdit())
        config.update_interval_s = std::clamp(config.update_interval_s,
                                              kMinUpdateIntervalS, kMaxUpdateIntervalS);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%.2f - %.0f s", kMinUpdateIntervalS, kMaxUpdateIntervalS);

    std::vector<DiscoveredController> found = discovered_controllers();
    if (!found.empty()) {
        for (size_t i = 0; i < found.size(); i++) {
            const DiscoveredController &c = found[i];
            std::string line = "Found " + (c.name.empty() ? std::string("WLED") : "\"" + c.name + "\"") +
                               " at " + c.ip;
            if (!c.mac.empty())
                line += "  (MAC " + c.mac + ")";
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(line.c_str());
            ImGui::SameLine();
            if (c.ip == config.controller_ip) {
                ImGui::TextDisabled("in use");
            } else {
                ImGui::PushID((int)i);
                if (ImGui::Button("Use")) {
                    config.controller_ip = c.ip;
                    std::snprintf(ip_buf, sizeof(ip_buf), "%s", c.ip.c_str());
                }
                ImGui::PopID();
            }
        }
    } else if (scanning) {
        ImGui::TextDisabled("%s", controller_scan_status().c_str());
    } else if (controller_listener_active()) {
        ImGui::TextDisabled("Listening for controllers (they announce about every 30 s)...");
    } else if (!controller_scan_status().empty()) {
        std::string st = controller_scan_status();
        ImGui::TextDisabled("%s%s", st.c_str(),
                            st.rfind("scanned", 0) == 0 ? ": no controller found" : "");
    }

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("W1:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputFloat("##W1", &config.cct_min_k, 0.0f, 0.0f, "%.0f");
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("W2:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputFloat("##W2", &config.cct_max_k, 0.0f, 0.0f, "%.0f");
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Cycle Length:");
    ImGui::SameLine();
    if (config.sync_to_real_time)
        ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(75.0f);
    ImGui::InputFloat("##CycleLength", &config.cycle_seconds, 0.0f, 0.0f,
                      "%.0f s");
    if (config.sync_to_real_time)
        ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Checkbox("Sync to Real Time", &config.sync_to_real_time);

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Zone:");
    ImGui::SameLine();
    if (ImGui::RadioButton("A", config.zone_mode == ZoneMode::AlwaysA))
        config.zone_mode = ZoneMode::AlwaysA;
    ImGui::SameLine();
    if (ImGui::RadioButton("B##zone", config.zone_mode == ZoneMode::AlwaysB))
        config.zone_mode = ZoneMode::AlwaysB;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scheduled", config.zone_mode == ZoneMode::Scheduled))
        config.zone_mode = ZoneMode::Scheduled;
    ImGui::SameLine();
    if (config.zone_mode != ZoneMode::Scheduled)
        ImGui::BeginDisabled();
    static char zone_start_buf[8], zone_end_buf[8];
    static float zone_start_shown = -1.0f, zone_end_shown = -1.0f;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("B from:");
    ImGui::SameLine();
    time_of_day_input("##ZoneBStart", config.zone_b_start, zone_start_buf,
                      zone_start_shown);
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("to:");
    ImGui::SameLine();
    time_of_day_input("##ZoneBEnd", config.zone_b_end, zone_end_buf, zone_end_shown);
    if (config.zone_mode != ZoneMode::Scheduled)
        ImGui::EndDisabled();
    if (running)
        ImGui::EndDisabled();

    static int selected_preset = 0;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Preset:");
    for (int i = 0; i < 5; i++) {
        ImGui::SameLine();
        if (ImGui::RadioButton(std::to_string(i + 1).c_str(),
                               selected_preset == i))
            selected_preset = i;
    }
    ImGui::SameLine();
    std::string preset_path =
        "lighting_preset_" + std::to_string(selected_preset + 1) + ".cfg";
    if (ImGui::Button("Save Preset"))
        save_lighting_config(config, preset_path);
    ImGui::SameLine();
    if (ImGui::Button("Load Preset")) {
        // A slot that's never been saved to has no file yet -- rather than
        // silently doing nothing, load falls back to the built-in default
        // lighting conditions, so every slot is always "loadable".
        if (!load_lighting_config(config, preset_path))
            config = LightingConfig();
        // ip_buf (the editable IP text box) only syncs into config.controller_ip
        // when Start Cycle is clicked, so it needs an explicit resync here or
        // a loaded/reset IP wouldn't actually show up in the box.
        std::snprintf(ip_buf, sizeof(ip_buf), "%s", config.controller_ip.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Editing curve:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Brightness", selected_curve == CurveId::Brightness))
        selected_curve = CurveId::Brightness;
    ImGui::SameLine();
    if (ImGui::RadioButton("CCT", selected_curve == CurveId::Cct))
        selected_curve = CurveId::Cct;
    ImGui::SameLine();
    if (ImGui::RadioButton("R", selected_curve == CurveId::R))
        selected_curve = CurveId::R;
    ImGui::SameLine();
    if (ImGui::RadioButton("G", selected_curve == CurveId::G))
        selected_curve = CurveId::G;
    ImGui::SameLine();
    if (ImGui::RadioButton("B", selected_curve == CurveId::B))
        selected_curve = CurveId::B;
    ImGui::SameLine();
    if (ImGui::Button("Reset Curve")) {
        bool is_white_curve = (selected_curve == CurveId::Brightness ||
                              selected_curve == CurveId::Cct);
        curve_ref(config, selected_curve) = is_white_curve
                                                ? make_default_bump_spline()
                                                : make_default_flat_spline();
    }

    // Right-justify the Start/Stop button on this same row.
    const char *cycle_btn_label = running ? "Stop Cycle" : "Start Cycle";
    float cycle_btn_w = ImGui::CalcTextSize(cycle_btn_label).x +
                        ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGui::SameLine(ImGui::GetWindowWidth() - cycle_btn_w -
                    ImGui::GetStyle().WindowPadding.x);
    if (!running) {
        if (ImGui::Button("Start Cycle"))
            start_lighting_thread(thread_ref, config);
    } else {
        if (ImGui::Button("Stop Cycle"))
            stop_lighting_thread(thread_ref);
        float cct = g_lighting_cct_k.load(std::memory_order_relaxed);
        float bri = g_lighting_brightness.load(std::memory_order_relaxed);
        float tf = g_lighting_time_frac.load(std::memory_order_relaxed);
        bool zone_b = g_lighting_zone_b.load(std::memory_order_relaxed);
        ImGui::Text("t=%.2f  CCT=%.0fK  brightness=%.0f%%  zone %s", tf, cct,
                   bri * 100.0f, zone_b ? "B" : "A");
        if (!g_lighting_link_ok.load(std::memory_order_relaxed)) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "controller not responding");
        }
    }

    ImGui::TextDisabled(
        "Drag points to reshape %s. Double-click to add a point, "
        "right-click a point to remove it.",
        curve_name(selected_curve));

    ImVec2 avail = ImGui::GetContentRegionAvail();
    // NoMenus: ImPlot's default right-click-to-open-settings context menu
    // would otherwise fight with right-click-to-remove-point below.
    if (ImPlot::BeginPlot("##lighting_preview", avail, ImPlotFlags_NoMenus)) {
        // NoButtons: ImPlot's legend swatches double as click-to-hide-series
        // toggles by default, which only hides that curve's *line* -- it does
        // nothing to the curve's data, the background swatch, or the actual
        // output to the controller. Left on, it looks exactly like a per-channel disable
        // switch but silently isn't one, so it's turned off here.
        ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
        bool editing_cct = (selected_curve == CurveId::Cct);
        ImPlot::SetupAxes(config.sync_to_real_time ? "time of day"
                                                   : "cycle position (0=start of day)",
                          editing_cct ? "CCT (K)" : "level (0-1)",
                          ImPlotAxisFlags_AutoFit, 0);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -kYAxisPadding, 1.0 + kYAxisPadding,
                                ImGuiCond_Always);
        // The underlying curve data is always normalized 0-1; while editing
        // CCT specifically, relabel the axis ticks in the units that curve
        // actually controls instead of the generic 0-1 scale. Likewise the
        // X axis reads as clock time instead of a 0-1 fraction whenever the
        // cycle is synced to the wall clock.
        if (editing_cct)
            ImPlot::SetupAxisFormat(ImAxis_Y1, format_cct_axis, &config);
        if (config.sync_to_real_time)
            ImPlot::SetupAxisFormat(ImAxis_X1, format_time_of_day_axis);

        draw_background_bands(config);
        draw_zone_strip(config);

        // Every curve renders as context; only the selected one gets drag
        // handles, so points from different curves can't be mixed up.
        plot_spline("brightness", config.brightness_curve,
                   curve_color(CurveId::Brightness),
                   selected_curve == CurveId::Brightness, 0);
        plot_spline("CCT", config.cct_curve, curve_color(CurveId::Cct),
                   selected_curve == CurveId::Cct, 100);
        plot_spline("R", config.r_curve, curve_color(CurveId::R),
                   selected_curve == CurveId::R, 200);
        plot_spline("G", config.g_curve, curve_color(CurveId::G),
                   selected_curve == CurveId::G, 300);
        plot_spline("B", config.b_curve, curve_color(CurveId::B),
                   selected_curve == CurveId::B, 400);

        if (running) {
            float tf = g_lighting_time_frac.load(std::memory_order_relaxed);
            ImPlot::PlotInfLines("current", &tf, 1);
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}
