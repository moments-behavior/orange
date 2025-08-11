#include "camera_manager.h"
#include "fetch_generated.h"
#include <chrono>
#include <filesystem>
#include <iostream>

namespace {
constexpr int kMaxCameras = 20;     // matches your earlier usage
constexpr int kEvtBufferSize = 100; // evt_buffer_size
} // namespace

// Start the manager thread
bool CameraManager::start(int *cam_count, ManagerContext *mgr,
                          GigEVisionDeviceInfo *unsorted,
                          GigEVisionDeviceInfo *sorted,
                          std::string *config_folder,
                          std::string *record_folder,
                          std::string *encoder_setup, PTPParams *ptp) {
    if (running_.load())
        return true;

    cam_count_ = cam_count;
    mgr_ = mgr;
    unsorted_ = unsorted;
    sorted_ = sorted;
    config_folder_ = config_folder;
    record_folder_ = record_folder;
    encoder_setup_ = encoder_setup;
    ptp_ = ptp;

    running_.store(true);
    thread_ = std::thread(&CameraManager::run_loop, this);
    return true;
}

void CameraManager::stop() {
    if (!running_.exchange(false))
        return;
    if (thread_.joinable())
        thread_.join();
}

void CameraManager::run_loop() {
    mgr_->state = FetchGame::ManagerState_IDLE;

    while (running_.load() && !mgr_->quit.load()) {
        switch (mgr_->state) {
        case FetchGame::ManagerState_CONNECT: {
            *cam_count_ = scan_cameras(kMaxCameras, unsorted_);
            sort_cameras_ip(unsorted_, sorted_, *cam_count_);
            mgr_->state = FetchGame::ManagerState_CONNECTED;
            break;
        }

        case FetchGame::ManagerState_OPENCAMERA: {
            // Allocate non-movable arrays (value-initialized)
            cams_count_ = *cam_count_;
            ecams_.reset(new CameraEmergent[cams_count_]{});
            cams_params_.reset(new CameraParams[cams_count_]{});
            cams_select_.reset(new CameraEachSelect[cams_count_]{});

            // Prepare configs and open each camera
            std::vector<std::string> cfgs;
            update_camera_configs(cfgs, *config_folder_);
            for (int i = 0; i < cams_count_; ++i) {
                set_camera_params(&cams_params_[i], &cams_select_[i],
                                  &sorted_[i], cfgs, i, cams_count_);
                open_camera_with_params(&ecams_[i].camera, &sorted_[i],
                                        &cams_params_[i]);
            }

            mgr_->state = FetchGame::ManagerState_CAMERAOPENED;
            break;
        }

        case FetchGame::ManagerState_STARTCAMTHREAD: {
            try {
                allocate_camera_frame_buffers(ecams_.get(), cams_params_.get(),
                                              kEvtBufferSize, cams_count_);
            } catch (...) {
                mgr_->state = FetchGame::ManagerState_ERROR;
                break;
            }

            cam_ctl_.record_video = true;
            cam_ctl_.subscribe = true;
            cam_ctl_.sync_camera = true;

            // Create output dir
            try {
                std::filesystem::create_directories(*record_folder_);
                std::cout << "Recorded video saves to: " << *record_folder_
                          << "\n";
            } catch (const std::exception &e) {
                std::cerr << "Error creating " << *record_folder_ << ": "
                          << e.what() << "\n";
                mgr_->state = FetchGame::ManagerState_ERROR;
                break;
            }

            // PTP sync all
            for (int i = 0; i < cams_count_; ++i) {
                ptp_camera_sync(&ecams_[i].camera, &cams_params_[i]);
            }

            // Ensure streams off initially
            for (int i = 0; i < cams_count_; ++i) {
                cams_select_[i].stream_on = false;
            }

            // Launch acquisition threads
            cam_threads_.clear();
            cam_threads_.reserve(static_cast<size_t>(cams_count_));
            for (int i = 0; i < cams_count_; ++i) {
                cam_threads_.emplace_back(acquire_frames, &ecams_[i],
                                          &cams_params_[i], &cams_select_[i],
                                          &cam_ctl_, nullptr, *encoder_setup_,
                                          *record_folder_, ptp_);
            }

            mgr_->state = FetchGame::ManagerState_THREADREADY;
            break;
        }

        default:
            break;
        } // switch state

        // Stop recording / teardown sequence
        if (ptp_->network_set_stop_ptp && ptp_->ptp_stop_reached) {
            ptp_->network_set_stop_ptp = false;

            // Join acquisition threads
            for (auto &t : cam_threads_)
                t.join();
            cam_threads_.clear();

            // Turn off PTP and close cameras, destroy buffers
            for (int i = 0; i < cams_count_; ++i) {
                ptp_sync_off(&ecams_[i].camera, &cams_params_[i]);
                destroy_frame_buffer(&ecams_[i].camera, ecams_[i].evt_frame,
                                     kEvtBufferSize, &cams_params_[i]);
                delete[] ecams_[i].evt_frame;
                check_camera_errors(EVT_CameraCloseStream(&ecams_[i].camera),
                                    cams_params_[i].camera_serial.c_str());
                close_camera(&ecams_[i].camera, &cams_params_[i]);
            }

            // Reset state
            *ptp_ = PTPParams{};
            cam_ctl_.sync_camera = false;

            // Release arrays (optional; they’ll be reused on next open)
            ecams_.reset();
            cams_params_.reset();
            cams_select_.reset();
            cams_count_ = 0;

            mgr_->state = FetchGame::ManagerState_RECORDSTOPPED;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Ensure threads are joined if we exit due to stop/quit
    for (auto &t : cam_threads_)
        if (t.joinable())
            t.join();
    cam_threads_.clear();
}
