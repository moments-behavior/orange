#pragma once
#include "camera.h"
#include "enet_runtime_select.h"
#include "fetch_generated.h"
#include "utils.h"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

struct ManagerContext {
    FetchGame::ManagerState state = FetchGame::ManagerState_IDLE;
    std::atomic<bool> quit{false};
};

class CameraManager {
  public:
    // Launches the manager thread
    bool start(int *cam_count, ManagerContext *mgr,
               GigEVisionDeviceInfo *unsorted, GigEVisionDeviceInfo *sorted,
               std::string *config_folder, std::string record_folder,
               std::string encoder_setup, PTPParams *ptp);

    // Joins the manager thread; safe to call multiple times
    void stop();

  private:
    void run_loop();

    // Inputs (non-owning)
    int *cam_count_ = nullptr;
    ManagerContext *mgr_ = nullptr;
    GigEVisionDeviceInfo *unsorted_ = nullptr;
    GigEVisionDeviceInfo *sorted_ = nullptr;
    std::string *config_folder_ = nullptr;
    PTPParams *ptp_ = nullptr;

    // Config
    std::string record_folder_;
    std::string encoder_setup_;

    // Resources — now as single, non-moving allocations
    int cams_count_ = 0;
    std::unique_ptr<CameraEmergent[]> ecams_;
    std::unique_ptr<CameraParams[]> cams_params_;
    std::unique_ptr<CameraEachSelect[]> cams_select_;
    CameraControl cam_ctl_{};

    std::vector<std::thread> cam_threads_; // threads are movable, it’s fine
    std::atomic<bool> running_{false};
    std::thread thread_;
};
