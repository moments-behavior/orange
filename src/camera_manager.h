// camera_manager.h
#pragma once
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
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
    std::string config_folder;
    int num_cameras = 0;
    GigEVisionDeviceInfo *device_info =
        nullptr; // lifetime owned by caller OR copy it in
};

struct StartArgs {
    RecordingSetup rec;
    PTPParams *ptp = nullptr; // or value-copy if that’s easier
};

enum class ManagerCmdType {
    ConnectScan,
    OpenCameras,
    StartThreads,
    StartRecording,
    StopRecording,
    Shutdown
};

struct ManagerCmd {
    ManagerCmdType type;
    // Only one of the following is used depending on type.
    OpenArgs open{};
    StartArgs start{};
    uint64_t ptp_time = 0; // for StartRecording/StopRecording
};

struct ManagerEvent {
    FetchGame::ManagerState state;
};

class CameraManager {
  public:
    using EventCallback = std::function<void(const ManagerEvent &)>;

    explicit CameraManager(EventCallback cb);
    ~CameraManager();

    // thread-safe command API
    void post(const ManagerCmd &cmd);

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

    // helpers
    void emit(FetchGame::ManagerState s);

  private:
    // owned by manager thread
    std::vector<CameraEmergent> ecams_;
    std::vector<CameraParams> cameras_params_;
    std::vector<CameraEachSelect> cameras_select_;
    std::vector<std::thread> camera_threads_;
    GigEVisionDeviceInfo unsorted_[max_cameras]{};
    GigEVisionDeviceInfo sorted_[max_cameras]{};
    int cam_count_ = 0;

    CameraControl camera_control_{};
    PTPParams *ptp_ = nullptr; // managed by caller (or make value-owned)
    RecordingSetup rec_{};

    // infra
    EventCallback cb_;
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<ManagerCmd> q_;
    bool stop_ = false;
};
