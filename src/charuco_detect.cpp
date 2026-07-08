#include "charuco_detect.h"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

const char *kDictNames[] = {
    "DICT_4X4_50",  "DICT_4X4_100",  "DICT_5X5_50",       "DICT_5X5_100",
    "DICT_5X5_250", "DICT_6X6_100",  "DICT_6X6_250",      "DICT_7X7_100",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_36h11",
};
const int kDictIds[] = {
    cv::aruco::DICT_4X4_50,  cv::aruco::DICT_4X4_100,
    cv::aruco::DICT_5X5_50,  cv::aruco::DICT_5X5_100,
    cv::aruco::DICT_5X5_250, cv::aruco::DICT_6X6_100,
    cv::aruco::DICT_6X6_250, cv::aruco::DICT_7X7_100,
    cv::aruco::DICT_APRILTAG_16h5, cv::aruco::DICT_APRILTAG_36h11,
};
constexpr int kNumDicts = sizeof(kDictIds) / sizeof(kDictIds[0]);

int dictionary_id(const std::string &name) {
    for (int i = 0; i < kNumDicts; i++) {
        if (name == kDictNames[i]) {
            return kDictIds[i];
        }
    }
    std::cerr << "charuco: unknown dictionary '" << name
              << "', falling back to DICT_5X5_100" << std::endl;
    return cv::aruco::DICT_5X5_100;
}

cv::aruco::CharucoBoard make_board(const CharucoBoardSpec &spec) {
    return cv::aruco::CharucoBoard(
        cv::Size(spec.squares_x, spec.squares_y), spec.square_mm,
        spec.marker_mm, cv::aruco::getPredefinedDictionary(dictionary_id(
                            spec.dictionary)));
}

} // namespace

const char *const *charuco_dictionary_names(int *count) {
    *count = kNumDicts;
    return kDictNames;
}

bool charuco_board_spec_load(const std::string &path, CharucoBoardSpec *spec) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    try {
        nlohmann::json j = nlohmann::json::parse(f);
        CharucoBoardSpec s;
        s.squares_x = j.value("squares_x", s.squares_x);
        s.squares_y = j.value("squares_y", s.squares_y);
        s.square_mm = j.value("square_mm", s.square_mm);
        s.marker_mm = j.value("marker_mm", s.marker_mm);
        s.dictionary = j.value("dictionary", s.dictionary);
        *spec = s;
        return true;
    } catch (const std::exception &e) {
        std::cerr << "charuco: failed to parse " << path << ": " << e.what()
                  << std::endl;
        return false;
    }
}

bool charuco_board_spec_save(const std::string &path,
                             const CharucoBoardSpec &spec) {
    nlohmann::json j;
    j["squares_x"] = spec.squares_x;
    j["squares_y"] = spec.squares_y;
    j["square_mm"] = spec.square_mm;
    j["marker_mm"] = spec.marker_mm;
    j["dictionary"] = spec.dictionary;
    std::ofstream f(path);
    if (!f.is_open()) {
        return false;
    }
    f << j.dump(4) << std::endl;
    return true;
}

CharucoBoardDetector::CharucoBoardDetector(const CharucoBoardSpec &spec)
    : spec_(spec), board_(make_board(spec)), detector_(board_) {}

CharucoDetection CharucoBoardDetector::detect(const cv::Mat &image) const {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    CharucoDetection det;
    try {
        detector_.detectBoard(gray, det.corners, det.ids, det.marker_corners,
                              det.marker_ids);
    } catch (const cv::Exception &e) {
        std::cerr << "charuco: detectBoard failed: " << e.what() << std::endl;
        det = CharucoDetection();
    }
    return det;
}

bool CharucoBoardDetector::pose(const CharucoDetection &det, const cv::Mat &k,
                                const cv::Mat &dist_coeffs, cv::Mat &rvec,
                                cv::Mat &tvec) const {
    if (det.num() < 4) {
        return false;
    }
    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    match_points(det, obj, img);
    if (obj.size() < 4) {
        return false;
    }
    // corners are coplanar (z=0) -> IPPE planar solver
    return cv::solvePnP(obj, img, k, dist_coeffs, rvec, tvec, false,
                        cv::SOLVEPNP_IPPE);
}

void CharucoBoardDetector::match_points(const CharucoDetection &det,
                                        std::vector<cv::Point3f> &obj,
                                        std::vector<cv::Point2f> &img) const {
    obj.clear();
    img.clear();
    const std::vector<cv::Point3f> board_corners =
        board_.getChessboardCorners();
    for (int i = 0; i < det.num(); i++) {
        int id = det.ids[i];
        if (id >= 0 && id < static_cast<int>(board_corners.size())) {
            obj.push_back(board_corners[id]);
            img.push_back(det.corners[i]);
        }
    }
}

void CharucoBoardDetector::board_image_points(
    const CharucoDetection &det, std::vector<cv::Point2f> &board_xy,
    std::vector<cv::Point2f> &img) const {
    board_xy.clear();
    img.clear();
    const std::vector<cv::Point3f> corners = board_.getChessboardCorners();
    for (int i = 0; i < det.num(); i++) {
        int id = det.ids[i];
        if (id >= 0 && id < static_cast<int>(corners.size())) {
            board_xy.emplace_back(corners[id].x, corners[id].y);
            img.push_back(det.corners[i]);
        }
    }
    // marker corners: board-frame quads live in getObjPoints(), indexed like
    // getIds()
    const std::vector<std::vector<cv::Point3f>> &obj = board_.getObjPoints();
    const std::vector<int> &ids = board_.getIds();
    for (size_t m = 0; m < det.marker_ids.size(); m++) {
        int idx = -1;
        for (size_t k = 0; k < ids.size(); k++) {
            if (ids[k] == det.marker_ids[m]) {
                idx = static_cast<int>(k);
                break;
            }
        }
        if (idx < 0 || det.marker_corners[m].size() != 4 ||
            obj[idx].size() != 4) {
            continue;
        }
        for (int c = 0; c < 4; c++) {
            board_xy.emplace_back(obj[idx][c].x, obj[idx][c].y);
            img.push_back(det.marker_corners[m][c]);
        }
    }
}

cv::Point3f CharucoBoardDetector::board_center() const {
    return cv::Point3f(0.5f * spec_.squares_x * spec_.square_mm,
                       0.5f * spec_.squares_y * spec_.square_mm, 0.0f);
}

bool CharucoBoardDetector::on_board(const cv::Point3f &board_pt,
                                    float margin_squares) const {
    float m = margin_squares * spec_.square_mm;
    return board_pt.x >= -m &&
           board_pt.x <= spec_.squares_x * spec_.square_mm + m &&
           board_pt.y >= -m &&
           board_pt.y <= spec_.squares_y * spec_.square_mm + m;
}
