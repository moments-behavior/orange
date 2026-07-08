#ifndef ORANGE_CHARUCO_DETECT
#define ORANGE_CHARUCO_DETECT

#include <opencv2/core.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <string>
#include <vector>

// ChArUco board detection for galvo calibration (docs/galvo_calibration_plan.md).
// ChArUco rather than plain ArUco because the zoomed-in galvo camera usually
// sees only part of the board; chessboard corners stay identifiable from
// partial views.

struct CharucoBoardSpec {
    int squares_x = 10;
    int squares_y = 7;
    float square_mm = 30.0f;
    float marker_mm = 22.0f;
    std::string dictionary = "DICT_5X5_100";
};

// JSON persistence (config/galvo_board.json). Load returns false and leaves
// *spec untouched if the file is missing/unreadable.
bool charuco_board_spec_load(const std::string &path, CharucoBoardSpec *spec);
bool charuco_board_spec_save(const std::string &path,
                             const CharucoBoardSpec &spec);
const char *const *charuco_dictionary_names(int *count);

struct CharucoDetection {
    std::vector<cv::Point2f> corners; // chessboard corners, px
    std::vector<int> ids;             // matching corner ids
    std::vector<std::vector<cv::Point2f>> marker_corners; // raw ArUco quads
    std::vector<int> marker_ids;
    int num() const { return static_cast<int>(corners.size()); }
    // total board<->image correspondences (chessboard + marker corners) --
    // a zoomed-in view holding a single marker still yields 4
    int num_points() const {
        return num() + 4 * static_cast<int>(marker_ids.size());
    }
};

class CharucoBoardDetector {
  public:
    explicit CharucoBoardDetector(const CharucoBoardSpec &spec);

    // Accepts gray or BGR input.
    CharucoDetection detect(const cv::Mat &image) const;

    // Board pose in the camera frame (X_cam = R*X_board + t) from >= 4
    // detected corners.
    bool pose(const CharucoDetection &det, const cv::Mat &k,
              const cv::Mat &dist_coeffs, cv::Mat &rvec, cv::Mat &tvec) const;

    // Object/image point pairs (chessboard corners only, 3D board frame).
    void match_points(const CharucoDetection &det,
                      std::vector<cv::Point3f> &obj,
                      std::vector<cv::Point2f> &img) const;

    // All board-plane (z=0, xy in mm) <-> image correspondences: chessboard
    // corners plus the raw marker corners. Enough for a local homography even
    // when only one marker fits in a zoomed view.
    void board_image_points(const CharucoDetection &det,
                            std::vector<cv::Point2f> &board_xy,
                            std::vector<cv::Point2f> &img) const;

    // Board-frame center (z=0 plane), mm.
    cv::Point3f board_center() const;
    // True if a board-frame point (z ~ 0) lies on the physical board,
    // extended by margin_squares squares (the board plane is exact slightly
    // past the physical edge; extrapolation error grows beyond that).
    bool on_board(const cv::Point3f &board_pt,
                  float margin_squares = 0.0f) const;

    const CharucoBoardSpec &spec() const { return spec_; }

  private:
    CharucoBoardSpec spec_;
    cv::aruco::CharucoBoard board_;
    cv::aruco::CharucoDetector detector_;
};

#endif
