#ifndef ORANGE_GALVO_CONTROL_LINK
#define ORANGE_GALVO_CONTROL_LINK

#include <cstdint>
#include <string>

// Request/reply UDP control channel to the GalvoControl app (GCC1 commands,
// GCS1 status replies -- see docs/galvo_calibration_plan.md and GalvoControl's
// PROTOCOL.md). Complements the one-way GCT1 target stream in galvo_sender.h:
// this channel commands raw mirror angles, toggles calib mode, and uploads
// fitted calibration parameters.
struct GalvoLinkParams {
    bool enabled = false;
    std::string target_ip = "10.102.10.90";
    int target_port = 5006;
};

// Decoded GCS1 reply. Angles are in the displayed motor frame (degrees).
struct GalvoStatus {
    bool ok = false;
    bool in_position = false;
    bool calib_mode = false;
    bool motors_enabled = false;
    bool remote_allowed = false;
    int err = 0; // 0 ok, 1 bad args, 2 remote disabled, 3 not connected, 4 clamped
    double pan_deg = 0.0;
    double tilt_deg = 0.0;
    double pan_min = 0.0, pan_max = 0.0;
    double tilt_min = 0.0, tilt_max = 0.0;
};

// The targeting model GalvoControl applies at runtime:
// v = R(rot)^T * (target - base); az/el of v -> sign*angle*scale + offset.
// rot is Euler XYZ degrees, R = Rz*Ry*Rx. Uploaded with galvo_link_set_calib.
struct GalvoCalibModel {
    double base[3] = {0.0, 0.0, 0.0};
    double rot[3] = {0.0, 0.0, 0.0};
    double pan_sign = 1.0, pan_scale = 0.5, pan_offset = 0.0;
    double tilt_sign = 1.0, tilt_scale = 0.5, tilt_offset = 0.0;
};

// Opens the socket and starts a 2 Hz status poller thread (the poll doubles
// as the calib-mode keepalive on the receiver). Safe to call again to
// re-point the link.
bool galvo_link_start(const std::string &target_ip, int target_port);
void galvo_link_stop();
bool galvo_link_running();

// Latest status captured by the poller. Returns false if the link is down or
// no reply has arrived yet. age_s (optional) is seconds since the last good
// reply -- the GUI treats > ~2s as link lost.
bool galvo_link_last_status(GalvoStatus *out, double *age_s = nullptr);

// Synchronous request/reply commands. Each blocks up to ~1s (250ms timeout,
// 4 tries) and returns true only on a validated GCS1 reply with err == 0
// (reply details in *out if given, including on error replies). Commands are
// idempotent so timeout retries are safe. Angles are displayed motor degrees.
bool galvo_link_ping(GalvoStatus *out = nullptr);
bool galvo_link_set_angles(double pan_deg, double tilt_deg,
                           GalvoStatus *out = nullptr);
bool galvo_link_calib_mode(bool on, GalvoStatus *out = nullptr);
bool galvo_link_set_calib(const GalvoCalibModel &model,
                          GalvoStatus *out = nullptr);
bool galvo_link_save_config(GalvoStatus *out = nullptr);
bool galvo_link_stop_motion(GalvoStatus *out = nullptr);

#endif
