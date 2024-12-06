#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include "imgui.h"
#include "camera_manager.h"
#include "camera_params.h"
#include "camera_preview_window.h"  // Changed from forward declaration to include

struct GigEVisionDeviceInfo;

class CameraControlPanel {
public:
    explicit CameraControlPanel(evt::CameraManager& camera_mgr);

    void setKnownCameras(const std::unordered_map<std::string, evt::CameraParams>& cameras) {
        known_cameras_ = &cameras;
        LOG(INFO) << "Set known cameras: " << cameras.size() << " configurations";
    }
    
    void render();
    
    void setDeviceInfo(const std::vector<GigEVisionDeviceInfo>& devices) {
        device_info = &devices;
        LOG(INFO) << "Set device info: " << devices.size() << " devices";
    }
    
    void setSelectedCameras(std::vector<bool>& selected) {
        selected_cameras = &selected;
        LOG(INFO) << "Set selected cameras vector of size: " << selected.size();
    }

    // Add state validation method
    bool isInitialized() const {
        return device_info && selected_cameras && known_cameras_ && 
               device_info->size() > 0;
    }

    // Add validation methods
    bool validateCameraAccess(size_t index) const {
        if (!device_info || !selected_cameras || !known_cameras_) {
            return false;
        }
        if (index >= device_info->size()) {
            return false;
        }
        return true;
    }

    bool isCameraSelected(size_t index) const {
        return validateCameraAccess(index) && (*selected_cameras)[index];
    }

    // Add preview window reference
    void setPreviewWindow(CameraPreviewWindow* preview) { preview_window_ = preview; }

private:
    void renderCameraList();
    void renderCameraSettings();
    void renderStreamingControls();
    void renderRecordingControls(const std::string& output_folder,
                                const std::string& encoder_setup);

    // Add helper methods
    void handleCameraError(const std::string& operation, const std::exception& e);
    bool ensureValidCameraState() const;
    void updateCameraStatus(size_t index, bool is_selected);

    // Add this method declaration
    void updateCameraParameter(size_t camera_idx, 
                             const std::string& serial,
                             const std::string& param_name, 
                             int value,
                             evt::CameraParams& params);

    // Data members for the panel
    evt::CameraManager& camera_manager;
    const std::vector<GigEVisionDeviceInfo>* device_info{nullptr};
    std::vector<bool>* selected_cameras{nullptr};
    const std::unordered_map<std::string, evt::CameraParams>* known_cameras_{nullptr};
    bool show_settings{false};
    bool show_temperature{false};
    CameraPreviewWindow* preview_window_{nullptr};
};