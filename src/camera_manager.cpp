// camera_manager.cpp
#include "camera_manager.h"
#include <chrono>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <system_error>

CameraManager::CameraManager(EventCallback cb) : cb_(std::move(cb)) {
    th_ = std::thread(&CameraManager::run, this);
}
CameraManager::~CameraManager() {
    post({ManagerCmdType::Shutdown});
    if (th_.joinable())
        th_.join();
}

void CameraManager::post(const ManagerCmd &cmd) {
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push(cmd);
    }
    cv_.notify_one();
}

void CameraManager::emit(FetchGame::ManagerState s) {
    if (cb_)
        cb_({s});
}

void CameraManager::run() {
    emit(FetchGame::ManagerState_IDLE);
    while (true) {
        ManagerCmd cmd;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return !q_.empty(); });
            cmd = q_.front();
            q_.pop();
        }
        switch (cmd.type) {
        case ManagerCmdType::ConnectScan:
            do_connect_scan();
            break;
        case ManagerCmdType::OpenCameras:
            do_open(cmd.open);
            break;
        case ManagerCmdType::StartThreads:
            do_start_threads(cmd.start);
            break;
        case ManagerCmdType::StartRecording:
            do_start_recording(cmd.ptp_time);
            break;
        case ManagerCmdType::StopRecording:
            do_stop_recording(cmd.ptp_time);
            break;
        case ManagerCmdType::Shutdown:
            do_shutdown();
            return;
        }
    }
}

void CameraManager::do_connect_scan() {
    cam_count_ = scan_cameras(max_cameras, unsorted_);
    sort_cameras_ip(unsorted_, sorted_, cam_count_);
    emit(FetchGame::ManagerState_CONNECTED);
}

void CameraManager::do_open(const OpenArgs &args) {
    // prepare vectors sized to camera count
    ecams_.assign(args.num_cameras, {});
    cameras_params_.assign(args.num_cameras, {});
    cameras_select_.assign(args.num_cameras, {});

    // load configs & open
    std::vector<std::string> camera_config_files;
    update_camera_configs(camera_config_files, args.config_folder);

    for (int i = 0; i < args.num_cameras; ++i) {
        set_camera_params(&cameras_params_[i], &cameras_select_[i],
                          &args.device_info[i], camera_config_files, i,
                          args.num_cameras);
        open_camera_with_params(&ecams_[i].camera,
                                &args.device_info[cameras_params_[i].camera_id],
                                &cameras_params_[i]);
    }
    emit(FetchGame::ManagerState_CAMERAOPENED);
}

void CameraManager::do_start_threads(const StartArgs &args) {
    rec_ = args.rec;
    ptp_ = args.ptp;

    // buffers
    allocate_camera_frame_buffers(ecams_.data(), cameras_params_.data(),
                                  evt_buffer_size, (int)ecams_.size());

    // mkdirs
    std::error_code ec;
    std::filesystem::create_directories(rec_.record_folder, ec);
    if (ec) {
        emit(FetchGame::ManagerState_ERROR);
        return;
    }
    std::cout << "Recorded video saves to: " << rec_.record_folder << "\n";

    camera_control_.record_video = true;
    camera_control_.subscribe = true;
    camera_control_.sync_camera = true;

    // sync setup
    for (size_t i = 0; i < ecams_.size(); ++i)
        ptp_camera_sync(&ecams_[i].camera, &cameras_params_[i]);

    // ensure select flags are off
    for (size_t i = 0; i < cameras_select_.size(); ++i)
        cameras_select_[i].stream_on = false;

    // launch threads
    camera_threads_.reserve(ecams_.size());
    for (size_t i = 0; i < ecams_.size(); ++i) {
        camera_threads_.emplace_back(
            &acquire_frames, &ecams_[i], &cameras_params_[i],
            &cameras_select_[i], &camera_control_, nullptr,
            rec_.encoder_basic_setup, rec_.record_folder, ptp_,
            std::ref(*get_app_context()) // or pass in via ctor
        );
    }

    // wait for all cameras ready (convert to CV if you can signal from threads)
    while (ptp_->ptp_counter != (int)ecams_.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    emit(FetchGame::ManagerState_THREADREADY);
}

void CameraManager::do_start_recording(uint64_t ptp_time) {
    if (!ptp_) {
        emit(FetchGame::ManagerState_ERROR);
        return;
    }
    ptp_->ptp_global_time = ptp_time;
    ptp_->network_set_start_ptp = true;
    emit(FetchGame::ManagerState_WAITSTOP);
}

void CameraManager::do_stop_recording(uint64_t ptp_time) {
    if (!ptp_) {
        emit(FetchGame::ManagerState_ERROR);
        return;
    }
    ptp_->ptp_stop_time = ptp_time;
    ptp_->network_set_stop_ptp = true;

    // wait for stop reach; ideally use CV signaled by camera threads
    while (!(ptp_->network_set_stop_ptp && ptp_->ptp_stop_reached)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // join threads
    for (auto &t : camera_threads_)
        t.join();
    camera_threads_.clear();

    // turn off sync + cleanup
    for (size_t i = 0; i < ecams_.size(); ++i) {
        ptp_sync_off(&ecams_[i].camera, &cameras_params_[i]);
        destroy_frame_buffer(&ecams_[i].camera, ecams_[i].evt_frame,
                             evt_buffer_size, &cameras_params_[i]);
        delete[] ecams_[i].evt_frame;
        check_camera_errors(EVT_CameraCloseStream(&ecams_[i].camera),
                            cameras_params_[i].camera_serial.c_str());
        close_camera(&ecams_[i].camera, &cameras_params_[i]);
    }
    ecams_.clear();
    cameras_params_.clear();
    cameras_select_.clear();

    // reset PTP flags
    *ptp_ = PTPParams{0, 0, 0, 0, true, false, false, false};

    camera_control_.sync_camera = false;

    emit(FetchGame::ManagerState_RECORDSTOPPED);
}

void CameraManager::do_shutdown() {
    // Best-effort cleanup if still running
    for (auto &t : camera_threads_)
        if (t.joinable())
            t.join();
    camera_threads_.clear();
    emit(FetchGame::ManagerState_IDLE);
}
