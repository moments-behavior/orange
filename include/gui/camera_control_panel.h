#pragma once

#include <memory>
#include "imgui.h"
#include "camera_manager.h"
#include "camera_params.h"

class CameraControlPanel {
public:
    explicit CameraControlPanel(evt::CameraManager& camera_mgr);
    
    void render();
    
    void setDeviceInfo(const std::vector<GigEVisionDeviceInfo>& devices) {
        device_info = &devices;
    }
    
    void setSelectedCameras(std::vector<bool>& selected) {
        selected_cameras = &selected;
    }

private:
    void renderCameraList();
    void renderCameraSettings();
    void renderStreamingControls();
    void renderRecordingControls(const std::string& output_folder,
                                const std::string& encoder_setup);

    // Data members for the panel
    evt::CameraManager& camera_manager;
    const std::vector<GigEVisionDeviceInfo>* device_info{nullptr};
    std::vector<bool>* selected_cameras{nullptr};
    bool show_settings{false};
    bool show_temperature{false};
};