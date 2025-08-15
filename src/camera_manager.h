// camera_manager.h
#pragma once
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory> // unique_ptr
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "fetch_game.h"
#include "utils.h"
#include "video_capture.h"

#define evt_buffer_size 100
#define max_cameras 20

struct RecordingSetup {
    std::string record_folder;
    std::string encoder_basic_setup;
};

struct OpenArgs {
    std::string folder;
};

struct StartArgs {
    RecordingSetup rec;
    PTPParams *ptp = nullptr; // managed by caller (or make value-owned)
};

enum class ManagerCmdType {
    ConnectScan,
    OpenCameras,
    StartThreads,
    StartRecording,
    StopRecording,
    Shutdown,
    StartCalib
};

struct ManagerCmd {
    ManagerCmdType type;
    OpenArgs open{};       // used for OpenCameras
    StartArgs start{};     // used for StartThreads
    uint64_t ptp_time = 0; // used for StartRecording/StopRecording
};

struct ManagerEvent {
    FetchGame::ManagerState state;
};

class CameraManager {
  public:
    using EventCallback = std::function<void(const ManagerEvent &)>;

    explicit CameraManager(AppContext &ctx, EventCallback cb);
    ~CameraManager();

    // thread-safe command API
    void post(const ManagerCmd &cmd);
    int cam_count() const noexcept { return cam_count_; }

  private:
    // manager thread
    void run();

    // impl steps (called only on manager thread)
    void do_connect_scan();
    void do_open(const OpenArgs &args);
    void do_start_threads(const StartArgs &args);
    void do_start_recording(uint64_t ptp_time);
    void do_stop_recording(uint64_t ptp_time);
    void do_shutdown();
    void do_start_calib(const OpenArgs &args);

    // helpers
    void emit(FetchGame::ManagerState s);

  private:
    // --- owned by manager thread ---
    AppContext &ctx_;
    std::string calib_folder_;
    // Contiguous, non-movable buffers for camera structures
    std::unique_ptr<CameraEmergent[]> ecams_;
    std::unique_ptr<CameraParams[]> cameras_params_;
    std::unique_ptr<CameraEachSelect[]> cameras_select_;
    size_t cam_n_ = 0;

    std::vector<std::thread> camera_threads_; // threads can live in a vector
    GigEVisionDeviceInfo unsorted_[max_cameras]{};
    GigEVisionDeviceInfo sorted_[max_cameras]{};
    int cam_count_ = 0;

    CameraControl camera_control_{};
    PTPParams *ptp_ = nullptr; // managed by caller (or make value-owned)
    RecordingSetup rec_{};

    // --- infra ---
    EventCallback cb_;
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<ManagerCmd> q_;
    bool stop_ = false;
};
