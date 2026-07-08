#ifndef ORANGE_GALVO_CALIB
#define ORANGE_GALVO_CALIB

#include "charuco_detect.h"
#include "galvo_control_link.h"
#include "video_capture.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Galvo <-> world calibration with a ChArUco board
// (docs/galvo_calibration_plan.md). A worker thread steps the mirrors through
// a pan/tilt grid over the control link; at each pose the board is detected in
// the galvo camera (which world point is the galvo gazing at?) and in a fixed
// calibrated camera (where is the board in world?). The samples are fit to
// the receiver's runtime targeting model and uploaded with
// galvo_link_set_calib.
//
// The galvo camera view is far too zoomed to hold the whole board -- often a
// single ArUco marker barely fits. So no pose and no galvo-camera intrinsics
// are used on that side: any >= 4 detected board points (one marker is
// enough) give a local homography, and the board-plane point under the image
// center is the gaze sample. The constant offset between image center and
// the true optical axis is identical at every sample, so the fitted pan/tilt
// offsets absorb it exactly.
//
// Requires SET_CALIB to reset the receiver's coordinate-frame mapping to
// identity: the fitted rotation IS the world->galvo alignment.

struct GalvoCalibConfig {
    // cameras (as passed around orange.cpp)
    int galvo_cam = -1; // index of the mirror-steered camera
    int num_cameras = 0;
    CameraEachSelect *cameras_select = nullptr;
    CameraParams *cameras_params = nullptr;

    CharucoBoardSpec board;

    // sweep: a grid centered on the zeroed/home pose (0,0) -- the useful
    // optical range around "facing forward" is far narrower than the travel
    // limits (which exist to protect the mechanics, not to bound where the
    // camera can actually see). Still clamped to the queried travel limits.
    int grid_pan = 7;
    int grid_tilt = 5;
    double pan_half_range_deg = 20.0;  // sweep pan in [-range, +range]
    double tilt_half_range_deg = 15.0; // sweep tilt in [-range, +range]
    double margin_deg = 3.0;           // keep inside the queried travel limits
    int placements_target = 2;    // board positions (>= 2 depths for a stable fit)
    int min_points_galvo = 4;     // board<->image points in the zoomed view
    int min_corners_fixed = 8;
    float board_margin_squares = 1.0f; // gaze may extrapolate this far off-board

    std::string out_folder; // calib_yaml folder: galvo_calib.json + galvo_intrinsics.yaml
};

enum GalvoCalibPhase {
    GalvoCalib_Idle,
    GalvoCalib_Busy,      // one-shot job (fitting, verify)
    GalvoCalib_Sweep,     // stepping the grid
    GalvoCalib_WaitBoard, // prompting the user to move the board
    GalvoCalib_Done,      // fit finished, model valid
    GalvoCalib_Error,
};

struct GalvoCalibSample {
    double pan_deg = 0.0, tilt_deg = 0.0; // measured motor angles
    cv::Point3d world;                    // gaze point, world mm
    int placement = 0;
    int points_galvo = 0, corners_fixed = 0;
    int fixed_cam = -1;
};

// Snapshot for the GUI; copied under a mutex each frame.
struct GalvoCalibStatusView {
    GalvoCalibPhase phase = GalvoCalib_Idle;
    std::string message;
    int placement = 0;
    int grid_index = 0, grid_total = 0;
    int accepted_total = 0, accepted_this_placement = 0;
    std::vector<uint8_t> coverage; // per grid cell: 0 pending, 1 skipped, 2 accepted
    int grid_pan = 0, grid_tilt = 0;
    bool fit_valid = false;
    double fit_rms_deg = -1.0;
    // verify: distance between the aimed gaze point and the board center,
    // measured on the board plane
    double verify_err_mm = -1.0;
    GalvoCalibModel model;
};

// --- jobs (one at a time; each returns false if another job is running) ---

// Grab one frame from every streaming camera and report how much of the
// board is detected in each -- instant feedback that the board spec
// (squares, sizes, dictionary) matches the physical print.
bool galvo_calib_test_detection(const GalvoCalibConfig &cfg);

// The main sweep -> (placements) -> fit -> save pipeline.
bool galvo_calib_start_sweep(const GalvoCalibConfig &cfg);
void galvo_calib_next_placement();    // user moved the board
void galvo_calib_finish_collection(); // fit with the placements done so far
void galvo_calib_abort();

// Closed-loop check of whatever calibration is live on the receiver: streams
// the board center as a GCT1 target (galvo_sender must be running), then
// measures how far the center lands from the galvo image center.
bool galvo_calib_start_verify(const GalvoCalibConfig &cfg);

bool galvo_calib_busy();
GalvoCalibStatusView galvo_calib_status();

// Reload the last persisted fit so the wizard resumes after an app restart.
// Cheap; safe to call when the file is missing.
void galvo_calib_load_persisted(const GalvoCalibConfig &cfg);

// --- frame tap (called from FrameSaver's thread; see FrameSaver.cpp) ---
bool galvo_calib_frame_wanted(CameraEachSelect *select);
void galvo_calib_deliver_frame(CameraEachSelect *select, const cv::Mat &bgr);

#endif
