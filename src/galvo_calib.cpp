#include "galvo_calib.h"
#include "galvo_sender.h"
#include "global.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <thread>

namespace {

constexpr double kRad2Deg = 57.29577951308232;
constexpr double kDeg2Rad = 1.0 / kRad2Deg;

// ---------------------------------------------------------------------------
// shared state
// ---------------------------------------------------------------------------
std::mutex g_state_mutex;
GalvoCalibStatusView g_status;
std::vector<GalvoCalibSample> g_samples;

std::thread g_worker;
std::atomic<bool> g_worker_running{false};
std::atomic<bool> g_abort{false};
std::atomic<bool> g_user_next{false};
std::atomic<bool> g_user_finish{false};

// frame tap handshake with FrameSaver threads
std::mutex g_grab_mutex;
std::condition_variable g_grab_cv;
CameraEachSelect *g_grab_want = nullptr;
cv::Mat g_grab_result;
bool g_grab_done = false;

void set_status(GalvoCalibPhase phase, const std::string &message) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_status.phase = phase;
    g_status.message = message;
}

std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// Launch fn on the single worker thread. Fails if a job is already running.
template <class Fn> bool launch_job(Fn &&fn) {
    if (g_worker_running.exchange(true)) {
        return false;
    }
    if (g_worker.joinable()) {
        g_worker.join();
    }
    g_abort.store(false);
    g_user_next.store(false);
    g_user_finish.store(false);
    g_worker = std::thread([fn = std::forward<Fn>(fn)]() mutable {
        fn();
        g_worker_running.store(false);
    });
    return true;
}

// ---------------------------------------------------------------------------
// frame tap
// ---------------------------------------------------------------------------
bool grab_frame(const GalvoCalibConfig &cfg, int cam_idx, cv::Mat &out,
                int timeout_ms = 3000) {
    if (cam_idx < 0 || cam_idx >= cfg.num_cameras) {
        return false;
    }
    CameraEachSelect *sel = &cfg.cameras_select[cam_idx];
    if (!sel->stream_on) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_grab_mutex);
        g_grab_want = sel;
        g_grab_done = false;
    }
    // hand the request to the acquisition thread via the still-frame path;
    // only if the FrameSaver is idle (a pending "Save pictures" wins)
    PictureState expected = State_Frame_Idle;
    if (!sel->sigs->frame_save_state.compare_exchange_strong(
            expected, State_Copy_New_Frame)) {
        std::lock_guard<std::mutex> lock(g_grab_mutex);
        g_grab_want = nullptr;
        return false;
    }
    std::unique_lock<std::mutex> lock(g_grab_mutex);
    bool ok = g_grab_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [] { return g_grab_done; });
    g_grab_want = nullptr;
    if (ok) {
        out = g_grab_result;
    }
    return ok;
}

std::string calib_json_path(const GalvoCalibConfig &cfg) {
    return cfg.out_folder + "/galvo_calib.json";
}

// ---------------------------------------------------------------------------
// geometry: one accepted grid pose -> world gaze point
// ---------------------------------------------------------------------------

// Board-plane point under the galvo image center, via a local homography from
// whatever board points are visible (a single ArUco marker is enough). No
// galvo-camera intrinsics involved: the constant image-center vs optical-axis
// offset is identical at every sample and is absorbed by the fitted pan/tilt
// offsets.
bool gaze_board_point(const CharucoBoardDetector &det,
                      const CharucoDetection &d, const cv::Size &image_size,
                      float margin_squares, cv::Point3f &board_pt) {
    std::vector<cv::Point2f> board_xy, img;
    det.board_image_points(d, board_xy, img);
    if (board_xy.size() < 4) {
        return false;
    }
    cv::Mat h;
    try {
        h = cv::findHomography(img, board_xy,
                               board_xy.size() >= 12 ? cv::RANSAC : 0, 3.0);
    } catch (const cv::Exception &) {
        return false;
    }
    if (h.empty()) {
        return false;
    }
    std::vector<cv::Point2f> center{cv::Point2f(image_size.width * 0.5f,
                                                image_size.height * 0.5f)};
    std::vector<cv::Point2f> mapped;
    cv::perspectiveTransform(center, mapped, h);
    board_pt = cv::Point3f(mapped[0].x, mapped[0].y, 0.0f);
    // the plane is exact slightly past the physical edge; farther out the
    // extrapolated homography degrades
    return det.on_board(board_pt, margin_squares);
}

// Board-frame point -> world, through a fixed camera's board pose (rvec/tvec)
// and its world extrinsics (X_cam = R_w*X_world + t_w).
cv::Point3d board_to_world(const cv::Point3f &board_pt, const cv::Mat &rvec_f,
                           const cv::Mat &tvec_f,
                           const CameraCalibResults &world_calib) {
    cv::Mat rf;
    cv::Rodrigues(rvec_f, rf);
    cv::Mat b = (cv::Mat_<double>(3, 1) << board_pt.x, board_pt.y, board_pt.z);
    cv::Mat x_cam = rf * b + tvec_f;
    cv::Mat rw, tw;
    world_calib.r.convertTo(rw, CV_64F);
    world_calib.tvec.convertTo(tw, CV_64F);
    cv::Mat x_w = rw.t() * (x_cam - tw);
    return cv::Point3d(x_w.at<double>(0), x_w.at<double>(1),
                       x_w.at<double>(2));
}

// Detect the board in the best fixed camera (most corners). Returns index or -1.
int detect_board_fixed(const GalvoCalibConfig &cfg,
                       const CharucoBoardDetector &det, CharucoDetection &best,
                       cv::Mat &frame_out) {
    int best_cam = -1;
    for (int i = 0; i < cfg.num_cameras; i++) {
        if (i == cfg.galvo_cam || !cfg.cameras_select[i].stream_on ||
            detection2d == nullptr || !detection2d[i].has_calibration_results) {
            continue;
        }
        cv::Mat frame;
        if (!grab_frame(cfg, i, frame)) {
            continue;
        }
        CharucoDetection d = det.detect(frame);
        if (d.num() >= cfg.min_corners_fixed && d.num() > best.num()) {
            best = d;
            best_cam = i;
            frame_out = frame;
        }
    }
    return best_cam;
}

// ---------------------------------------------------------------------------
// model fit: multi-start Levenberg-Marquardt on the receiver's runtime model
//   v = R(rot)^T * (X - O);  az/el of v;  pan = az*ps + po;  tilt = el*ts + to
// params p[10] = {O(3) mm, rot(3) Euler XYZ deg, ps, po, ts, to}. Scales are
// signed here; split into sign+|scale| only when uploading.
// ---------------------------------------------------------------------------

void euler_to_r(const double rot_deg[3], double r[3][3]) {
    double rx = rot_deg[0] * kDeg2Rad, ry = rot_deg[1] * kDeg2Rad,
           rz = rot_deg[2] * kDeg2Rad;
    double cx = cos(rx), sx = sin(rx);
    double cy = cos(ry), sy = sin(ry);
    double cz = cos(rz), sz = sin(rz);
    // R = Rz*Ry*Rx (must match GalvoControl's world_to_azel_base)
    r[0][0] = cz * cy;
    r[0][1] = cz * sy * sx - sz * cx;
    r[0][2] = cz * sy * cx + sz * sx;
    r[1][0] = sz * cy;
    r[1][1] = sz * sy * sx + cz * cx;
    r[1][2] = sz * sy * cx - cz * sx;
    r[2][0] = -sy;
    r[2][1] = cy * sx;
    r[2][2] = cy * cx;
}

void model_azel(const double *p, const cv::Point3d &w, double &az,
                double &el) {
    double r[3][3];
    euler_to_r(p + 3, r);
    double d[3] = {w.x - p[0], w.y - p[1], w.z - p[2]};
    // v = R^T d
    double vx = r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2];
    double vy = r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2];
    double vz = r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2];
    az = atan2(vx, vz) * kRad2Deg;
    el = atan2(vy, sqrt(vx * vx + vz * vz)) * kRad2Deg;
}

void model_residuals(const double *p,
                     const std::vector<GalvoCalibSample> &samples,
                     std::vector<double> &res) {
    res.resize(samples.size() * 2);
    for (size_t i = 0; i < samples.size(); i++) {
        double az, el;
        model_azel(p, samples[i].world, az, el);
        res[2 * i] = (az * p[6] + p[7]) - samples[i].pan_deg;
        res[2 * i + 1] = (el * p[8] + p[9]) - samples[i].tilt_deg;
    }
}

double model_sse(const double *p,
                 const std::vector<GalvoCalibSample> &samples) {
    std::vector<double> res;
    model_residuals(p, samples, res);
    double s = 0.0;
    for (double r : res) {
        s += r * r;
    }
    return s;
}

// Closed-form init of the 4 linear params (scale/offset per axis) for fixed
// O/rot: simple 1D least squares pan ~ az, tilt ~ el.
void init_linear_params(double *p,
                        const std::vector<GalvoCalibSample> &samples) {
    double s_az = 0, s_el = 0, s_pan = 0, s_tilt = 0, s_azaz = 0, s_elel = 0,
           s_azpan = 0, s_eltilt = 0;
    int n = static_cast<int>(samples.size());
    for (const GalvoCalibSample &s : samples) {
        double az, el;
        model_azel(p, s.world, az, el);
        s_az += az;
        s_el += el;
        s_pan += s.pan_deg;
        s_tilt += s.tilt_deg;
        s_azaz += az * az;
        s_elel += el * el;
        s_azpan += az * s.pan_deg;
        s_eltilt += el * s.tilt_deg;
    }
    double var_az = s_azaz - s_az * s_az / n;
    double var_el = s_elel - s_el * s_el / n;
    p[6] = var_az > 1e-6 ? (s_azpan - s_az * s_pan / n) / var_az : 0.5;
    p[7] = (s_pan - p[6] * s_az) / n;
    p[8] = var_el > 1e-6 ? (s_eltilt - s_el * s_tilt / n) / var_el : 0.5;
    p[9] = (s_tilt - p[8] * s_el) / n;
}

// LM with numeric Jacobian. Param steps sized per unit (mm / deg / ratio).
double lm_refine(double *p, const std::vector<GalvoCalibSample> &samples,
                 int max_iters = 80) {
    const double steps[10] = {1.0,  1.0,  1.0,  0.05, 0.05,
                              0.05, 1e-4, 0.01, 1e-4, 0.01};
    int n = static_cast<int>(samples.size()) * 2;
    std::vector<double> res, res_try;
    model_residuals(p, samples, res);
    double sse = 0.0;
    for (double r : res) {
        sse += r * r;
    }
    double lambda = 1e-3;

    for (int iter = 0; iter < max_iters; iter++) {
        // numeric Jacobian, forward differences
        cv::Mat jac(n, 10, CV_64F);
        for (int j = 0; j < 10; j++) {
            double saved = p[j];
            p[j] += steps[j];
            model_residuals(p, samples, res_try);
            p[j] = saved;
            for (int i = 0; i < n; i++) {
                jac.at<double>(i, j) = (res_try[i] - res[i]) / steps[j];
            }
        }
        cv::Mat r_vec(n, 1, CV_64F, res.data());
        cv::Mat jtj = jac.t() * jac;
        cv::Mat jtr = jac.t() * r_vec;

        bool improved = false;
        for (int attempt = 0; attempt < 8 && !improved; attempt++) {
            cv::Mat a = jtj.clone();
            for (int j = 0; j < 10; j++) {
                a.at<double>(j, j) += lambda * (jtj.at<double>(j, j) + 1e-9);
            }
            cv::Mat delta;
            if (!cv::solve(a, jtr, delta, cv::DECOMP_SVD)) {
                lambda *= 10.0;
                continue;
            }
            double p_try[10];
            for (int j = 0; j < 10; j++) {
                p_try[j] = p[j] - delta.at<double>(j);
            }
            double sse_try = model_sse(p_try, samples);
            if (sse_try < sse) {
                memcpy(p, p_try, sizeof(p_try));
                sse = sse_try;
                model_residuals(p, samples, res);
                lambda = std::max(lambda * 0.3, 1e-9);
                improved = true;
            } else {
                lambda *= 10.0;
            }
        }
        if (!improved) {
            break; // converged (or stuck)
        }
    }
    return sse;
}

bool fit_model(const std::vector<GalvoCalibSample> &samples,
               GalvoCalibModel *model, double *rms_deg) {
    int n = static_cast<int>(samples.size());
    if (n < 8) {
        return false;
    }
    cv::Point3d c(0, 0, 0);
    cv::Point3d lo(1e30, 1e30, 1e30), hi(-1e30, -1e30, -1e30);
    for (const GalvoCalibSample &s : samples) {
        c += s.world;
        lo.x = std::min(lo.x, s.world.x);
        lo.y = std::min(lo.y, s.world.y);
        lo.z = std::min(lo.z, s.world.z);
        hi.x = std::max(hi.x, s.world.x);
        hi.y = std::max(hi.y, s.world.y);
        hi.z = std::max(hi.z, s.world.z);
    }
    c *= 1.0 / n;
    double diag = cv::norm(hi - lo);
    double dist = std::max(1000.0, 2.0 * diag);

    // Multi-start: the pivot is unknown, so try it out along each world axis
    // from the sample centroid, x each cardinal yaw/pitch of the galvo frame.
    // LM from the best-costed inits converges in well under a second.
    const double offsets[6][3] = {{dist, 0, 0},  {-dist, 0, 0}, {0, dist, 0},
                                  {0, -dist, 0}, {0, 0, dist},  {0, 0, -dist}};
    const double rots[6][3] = {{0, 0, 0},   {0, 90, 0}, {0, 180, 0},
                               {0, 270, 0}, {90, 0, 0}, {-90, 0, 0}};

    double best_p[10];
    double best_sse = 1e300;
    for (const auto &off : offsets) {
        for (const auto &rot : rots) {
            double p[10] = {c.x + off[0], c.y + off[1], c.z + off[2],
                            rot[0],       rot[1],       rot[2],
                            0.5,          0.0,          0.5,
                            0.0};
            init_linear_params(p, samples);
            double sse = lm_refine(p, samples, 60);
            if (sse < best_sse) {
                best_sse = sse;
                memcpy(best_p, p, sizeof(p));
            }
        }
    }
    if (best_sse >= 1e300) {
        return false;
    }
    // polish the winner
    best_sse = lm_refine(best_p, samples, 120);

    model->base[0] = best_p[0];
    model->base[1] = best_p[1];
    model->base[2] = best_p[2];
    // keep euler angles presentable
    for (int i = 0; i < 3; i++) {
        double a = std::fmod(best_p[3 + i], 360.0);
        if (a > 180.0) {
            a -= 360.0;
        } else if (a < -180.0) {
            a += 360.0;
        }
        model->rot[i] = a;
    }
    model->pan_sign = best_p[6] < 0 ? -1.0 : 1.0;
    model->pan_scale = std::fabs(best_p[6]);
    model->pan_offset = best_p[7];
    model->tilt_sign = best_p[8] < 0 ? -1.0 : 1.0;
    model->tilt_scale = std::fabs(best_p[8]);
    model->tilt_offset = best_p[9];
    *rms_deg = std::sqrt(best_sse / (2.0 * n));
    return true;
}

// ---------------------------------------------------------------------------
// persistence of samples + fit
// ---------------------------------------------------------------------------
void save_calib_json(const GalvoCalibConfig &cfg,
                     const std::vector<GalvoCalibSample> &samples,
                     const GalvoCalibModel &model, double rms, bool fit_valid) {
    nlohmann::json j;
    j["date"] = timestamp_now();
    j["board"] = {{"squares_x", cfg.board.squares_x},
                  {"squares_y", cfg.board.squares_y},
                  {"square_mm", cfg.board.square_mm},
                  {"marker_mm", cfg.board.marker_mm},
                  {"dictionary", cfg.board.dictionary}};
    j["samples"] = nlohmann::json::array();
    for (const GalvoCalibSample &s : samples) {
        j["samples"].push_back({{"pan", s.pan_deg},
                                {"tilt", s.tilt_deg},
                                {"x", s.world.x},
                                {"y", s.world.y},
                                {"z", s.world.z},
                                {"placement", s.placement},
                                {"points_galvo", s.points_galvo},
                                {"corners_fixed", s.corners_fixed},
                                {"fixed_cam", s.fixed_cam}});
    }
    if (fit_valid) {
        j["model"] = {{"base", {model.base[0], model.base[1], model.base[2]}},
                      {"rot", {model.rot[0], model.rot[1], model.rot[2]}},
                      {"pan_sign", model.pan_sign},
                      {"pan_scale", model.pan_scale},
                      {"pan_offset", model.pan_offset},
                      {"tilt_sign", model.tilt_sign},
                      {"tilt_scale", model.tilt_scale},
                      {"tilt_offset", model.tilt_offset},
                      {"rms_deg", rms}};
    }
    std::ofstream f(calib_json_path(cfg));
    if (f.is_open()) {
        f << j.dump(2) << std::endl;
    }
}

bool load_calib_json(const GalvoCalibConfig &cfg, GalvoCalibModel *model,
                     double *rms) {
    std::ifstream f(calib_json_path(cfg));
    if (!f.is_open()) {
        return false;
    }
    try {
        nlohmann::json j = nlohmann::json::parse(f);
        if (!j.contains("model")) {
            return false;
        }
        const auto &m = j["model"];
        for (int i = 0; i < 3; i++) {
            model->base[i] = m["base"][i];
            model->rot[i] = m["rot"][i];
        }
        model->pan_sign = m["pan_sign"];
        model->pan_scale = m["pan_scale"];
        model->pan_offset = m["pan_offset"];
        model->tilt_sign = m["tilt_sign"];
        model->tilt_scale = m["tilt_scale"];
        model->tilt_offset = m["tilt_offset"];
        *rms = m.value("rms_deg", -1.0);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "galvo_calib: failed to parse galvo_calib.json: "
                  << e.what() << std::endl;
        return false;
    }
}

// ---------------------------------------------------------------------------
// sweep worker
// ---------------------------------------------------------------------------

// Move the mirrors and wait until the receiver reports in-position near the
// commanded angles. Returns the measured angles.
bool move_and_settle(double pan_cmd, double tilt_cmd, double &pan_meas,
                     double &tilt_meas, std::string &err) {
    GalvoStatus rep;
    if (!galvo_link_set_angles(pan_cmd, tilt_cmd, &rep)) {
        err = rep.err == 2   ? "remote control disabled on the galvo app"
              : rep.err == 3 ? "galvo motors not connected"
              : rep.err == 4 ? "angle clamped (outside travel limits)"
                             : "no reply from the galvo app";
        return false;
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    // give the receiver a beat to leave in-position before trusting it
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    while (std::chrono::steady_clock::now() < deadline) {
        if (g_abort.load()) {
            err = "aborted";
            return false;
        }
        if (galvo_link_ping(&rep) && rep.in_position &&
            std::fabs(rep.pan_deg - pan_cmd) < 0.5 &&
            std::fabs(rep.tilt_deg - tilt_cmd) < 0.5) {
            // settle: let vibration die down
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (!galvo_link_ping(&rep)) {
                break;
            }
            pan_meas = rep.pan_deg;
            tilt_meas = rep.tilt_deg;
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    err = "timed out waiting for the mirrors to reach position";
    return false;
}

// Park the mirrors back at the zeroed/home pose (still in calib mode so the
// aim loop can't fight the move), then hand control back.
void sweep_end_motion() {
    galvo_link_set_angles(0.0, 0.0, nullptr);
    galvo_link_calib_mode(false, nullptr);
}

void sweep_worker(GalvoCalibConfig cfg) {
    CharucoBoardDetector detector(cfg.board);
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_samples.clear();
        g_status.fit_valid = false;
        g_status.fit_rms_deg = -1.0;
        g_status.accepted_total = 0;
        g_status.placement = 0;
    }

    GalvoStatus st;
    if (!galvo_link_ping(&st)) {
        set_status(GalvoCalib_Error, "galvo control link is not responding");
        return;
    }
    if (!st.remote_allowed) {
        set_status(GalvoCalib_Error,
                   "enable 'allow remote control' on the galvo app");
        return;
    }
    if (!galvo_link_calib_mode(true, nullptr)) {
        set_status(GalvoCalib_Error, "could not enter calib mode");
        return;
    }

    // grid centered on the zeroed/home pose (0,0), clamped to the travel
    // limits (serpentine order minimizes travel)
    double pan_lo = std::max(-cfg.pan_half_range_deg,
                             st.pan_min + cfg.margin_deg);
    double pan_hi = std::min(cfg.pan_half_range_deg,
                             st.pan_max - cfg.margin_deg);
    double tilt_lo = std::max(-cfg.tilt_half_range_deg,
                              st.tilt_min + cfg.margin_deg);
    double tilt_hi = std::min(cfg.tilt_half_range_deg,
                              st.tilt_max - cfg.margin_deg);
    if (pan_lo >= pan_hi || tilt_lo >= tilt_hi) {
        galvo_link_calib_mode(false, nullptr);
        set_status(GalvoCalib_Error,
                   "empty sweep range -- the home pose (0,0) must sit inside "
                   "the travel limits, and the sweep range must be > 0");
        return;
    }
    int grid_total = cfg.grid_pan * cfg.grid_tilt;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_status.grid_total = grid_total;
        g_status.grid_pan = cfg.grid_pan;
        g_status.grid_tilt = cfg.grid_tilt;
    }

    for (int placement = 0; placement < cfg.placements_target; placement++) {
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_status.placement = placement + 1;
            g_status.accepted_this_placement = 0;
            g_status.coverage.assign(grid_total, 0);
            g_status.phase = GalvoCalib_Sweep;
        }

        for (int it = 0; it < cfg.grid_tilt && !g_abort.load(); it++) {
            for (int ip_raw = 0; ip_raw < cfg.grid_pan && !g_abort.load();
                 ip_raw++) {
                int ip = (it % 2 == 0) ? ip_raw : cfg.grid_pan - 1 - ip_raw;
                int cell = it * cfg.grid_pan + ip;
                double pan_cmd =
                    pan_lo + (pan_hi - pan_lo) * ip /
                                 std::max(1, cfg.grid_pan - 1);
                double tilt_cmd =
                    tilt_lo + (tilt_hi - tilt_lo) * it /
                                  std::max(1, cfg.grid_tilt - 1);
                {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_status.grid_index = it * cfg.grid_pan + ip_raw + 1;
                    g_status.message =
                        "placement " + std::to_string(placement + 1) + ": pan " +
                        std::to_string(pan_cmd).substr(0, 6) + " tilt " +
                        std::to_string(tilt_cmd).substr(0, 6);
                }

                double pan_meas, tilt_meas;
                std::string err;
                if (!move_and_settle(pan_cmd, tilt_cmd, pan_meas, tilt_meas,
                                     err)) {
                    if (g_abort.load()) {
                        break;
                    }
                    sweep_end_motion();
                    set_status(GalvoCalib_Error, "sweep failed: " + err);
                    return;
                }

                auto mark = [&](uint8_t v) {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_status.coverage[cell] = v;
                };

                // does the galvo camera see the board here?
                cv::Mat galvo_frame;
                if (!grab_frame(cfg, cfg.galvo_cam, galvo_frame)) {
                    sweep_end_motion();
                    set_status(GalvoCalib_Error,
                               "could not grab a galvo camera frame -- is "
                               "streaming on?");
                    return;
                }
                CharucoDetection det_g = detector.detect(galvo_frame);
                cv::Point3f board_pt;
                if (det_g.num_points() < cfg.min_points_galvo ||
                    !gaze_board_point(detector, det_g, galvo_frame.size(),
                                      cfg.board_margin_squares, board_pt)) {
                    mark(1);
                    continue;
                }

                // where is that board point in the world?
                CharucoDetection det_f;
                cv::Mat fixed_frame;
                int fixed_cam = detect_board_fixed(cfg, detector, det_f,
                                                   fixed_frame);
                if (fixed_cam < 0) {
                    mark(1);
                    continue;
                }
                cv::Mat rvec_f, tvec_f;
                if (!detector.pose(det_f, detection2d[fixed_cam].camera_calib.k,
                                   detection2d[fixed_cam].camera_calib
                                       .dist_coeffs,
                                   rvec_f, tvec_f)) {
                    mark(1);
                    continue;
                }

                GalvoCalibSample sample;
                sample.pan_deg = pan_meas;
                sample.tilt_deg = tilt_meas;
                sample.world =
                    board_to_world(board_pt, rvec_f, tvec_f,
                                   detection2d[fixed_cam].camera_calib);
                sample.placement = placement;
                sample.points_galvo = det_g.num_points();
                sample.corners_fixed = det_f.num();
                sample.fixed_cam = fixed_cam;
                {
                    std::lock_guard<std::mutex> lock(g_state_mutex);
                    g_samples.push_back(sample);
                    g_status.accepted_total =
                        static_cast<int>(g_samples.size());
                    g_status.accepted_this_placement++;
                    g_status.coverage[cell] = 2;
                }
            }
        }
        if (g_abort.load() || g_user_finish.load()) {
            break;
        }

        if (placement + 1 < cfg.placements_target) {
            set_status(GalvoCalib_WaitBoard,
                       "move the board to a different distance, then press "
                       "'Board moved -- continue'");
            g_user_next.store(false);
            while (!g_user_next.load() && !g_user_finish.load() &&
                   !g_abort.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    sweep_end_motion();
    if (g_abort.load()) {
        set_status(GalvoCalib_Idle, "sweep aborted");
        return;
    }

    // fit + persist
    std::vector<GalvoCalibSample> samples;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        samples = g_samples;
    }
    set_status(GalvoCalib_Busy, "fitting " + std::to_string(samples.size()) +
                                    " samples...");
    GalvoCalibModel model;
    double rms = -1.0;
    bool ok = fit_model(samples, &model, &rms);
    save_calib_json(cfg, samples, model, rms, ok);
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_status.fit_valid = ok;
        g_status.fit_rms_deg = rms;
        if (ok) {
            g_status.model = model;
        }
    }
    if (!ok) {
        set_status(GalvoCalib_Error,
                   "fit failed -- need >= 8 accepted samples spanning both "
                   "axes (got " +
                       std::to_string(samples.size()) + ")");
        return;
    }
    set_status(GalvoCalib_Done,
               "fit OK: RMS " + std::to_string(rms).substr(0, 5) +
                   " deg over " + std::to_string(samples.size()) +
                   " samples -- review, then Upload");
}

// ---------------------------------------------------------------------------
// verify worker: closed loop through the live GCT1 path
// ---------------------------------------------------------------------------
void verify_worker(GalvoCalibConfig cfg) {
    CharucoBoardDetector detector(cfg.board);
    if (!galvo_sender_running()) {
        set_status(GalvoCalib_Error,
                   "enable galvo target streaming first (the verify target "
                   "is sent over the normal GCT1 path)");
        return;
    }
    // make sure normal aiming isn't suppressed
    galvo_link_calib_mode(false, nullptr);

    // board center in world, via the best fixed camera
    CharucoDetection det_f;
    cv::Mat fixed_frame;
    int fixed_cam = detect_board_fixed(cfg, detector, det_f, fixed_frame);
    if (fixed_cam < 0) {
        set_status(GalvoCalib_Error,
                   "board not visible in any calibrated fixed camera");
        return;
    }
    cv::Mat rvec_f, tvec_f;
    if (!detector.pose(det_f, detection2d[fixed_cam].camera_calib.k,
                       detection2d[fixed_cam].camera_calib.dist_coeffs, rvec_f,
                       tvec_f)) {
        set_status(GalvoCalib_Error, "board pose failed in the fixed camera");
        return;
    }
    cv::Point3d target = board_to_world(detector.board_center(), rvec_f,
                                        tvec_f,
                                        detection2d[fixed_cam].camera_calib);

    // stream the target and let the receiver aim (slew-limited, ~30 Hz)
    set_status(GalvoCalib_Busy, "aiming at the board center...");
    auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < until && !g_abort.load()) {
        galvo_sender_send_target(target.x, target.y, target.z, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    galvo_sender_send_target(target.x, target.y, target.z, false); // hold

    // where on the board plane is the galvo actually gazing now?
    cv::Mat galvo_frame;
    if (!grab_frame(cfg, cfg.galvo_cam, galvo_frame)) {
        set_status(GalvoCalib_Error, "could not grab a galvo camera frame");
        return;
    }
    CharucoDetection det_g = detector.detect(galvo_frame);
    cv::Point3f gaze;
    // generous margin: a large pointing error still yields a measurement as
    // long as some marker is in view
    if (det_g.num_points() < cfg.min_points_galvo ||
        !gaze_board_point(detector, det_g, galvo_frame.size(),
                          cfg.board_margin_squares + 4.0f, gaze)) {
        set_status(GalvoCalib_Error,
                   "board not visible in the galvo camera after aiming -- "
                   "pointing error is large or the board moved");
        return;
    }
    cv::Point3f c = detector.board_center();
    double err_mm = std::hypot(gaze.x - c.x, gaze.y - c.y);
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_status.verify_err_mm = err_mm;
    }
    set_status(GalvoCalib_Idle,
               "verify: gaze center lands " +
                   std::to_string(err_mm).substr(0, 6) +
                   " mm from the board center (on the board plane)");
}

} // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

bool galvo_calib_test_detection(const GalvoCalibConfig &cfg) {
    return launch_job([cfg] {
        set_status(GalvoCalib_Busy, "testing board detection...");
        CharucoBoardDetector detector(cfg.board);
        std::string report;
        int total = 0;
        for (int i = 0; i < cfg.num_cameras; i++) {
            if (!cfg.cameras_select[i].stream_on) {
                continue;
            }
            std::string name = cfg.cameras_params != nullptr
                                   ? cfg.cameras_params[i].camera_serial
                                   : std::to_string(i);
            if (i == cfg.galvo_cam) {
                name += " (galvo)";
            }
            cv::Mat frame;
            if (!grab_frame(cfg, i, frame)) {
                report += name + ": no frame; ";
                continue;
            }
            CharucoDetection d = detector.detect(frame);
            report += name + ": " + std::to_string(d.marker_ids.size()) +
                      " markers, " + std::to_string(d.num()) + " corners; ";
            total += d.num_points();
        }
        if (report.empty()) {
            set_status(GalvoCalib_Error,
                       "no cameras streaming -- start streaming first");
            return;
        }
        set_status(total > 0 ? GalvoCalib_Idle : GalvoCalib_Error,
                   total > 0
                       ? "detection: " + report
                       : "detection: " + report +
                             "-- nothing found: check the dictionary first, "
                             "then squares x/y");
    });
}

bool galvo_calib_start_sweep(const GalvoCalibConfig &cfg) {
    if (cfg.galvo_cam < 0 || cfg.galvo_cam >= cfg.num_cameras ||
        cfg.cameras_select == nullptr) {
        return false;
    }
    return launch_job([cfg] { sweep_worker(cfg); });
}

bool galvo_calib_start_verify(const GalvoCalibConfig &cfg) {
    if (cfg.galvo_cam < 0 || cfg.galvo_cam >= cfg.num_cameras ||
        cfg.cameras_select == nullptr) {
        return false;
    }
    return launch_job([cfg] { verify_worker(cfg); });
}

void galvo_calib_next_placement() { g_user_next.store(true); }
void galvo_calib_finish_collection() { g_user_finish.store(true); }
void galvo_calib_abort() { g_abort.store(true); }

bool galvo_calib_busy() { return g_worker_running.load(); }

GalvoCalibStatusView galvo_calib_status() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_status;
}

void galvo_calib_load_persisted(const GalvoCalibConfig &cfg) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!g_status.fit_valid) {
        GalvoCalibModel model;
        double rms;
        if (load_calib_json(cfg, &model, &rms)) {
            g_status.model = model;
            g_status.fit_rms_deg = rms;
            g_status.fit_valid = true;
        }
    }
}

bool galvo_calib_frame_wanted(CameraEachSelect *select) {
    std::lock_guard<std::mutex> lock(g_grab_mutex);
    return g_grab_want == select;
}

void galvo_calib_deliver_frame(CameraEachSelect *select, const cv::Mat &bgr) {
    std::lock_guard<std::mutex> lock(g_grab_mutex);
    if (g_grab_want != select) {
        return;
    }
    g_grab_result = bgr.clone();
    g_grab_done = true;
    g_grab_cv.notify_all();
}
