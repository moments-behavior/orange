#pragma once

#include <memory>
#include "imgui.h"
#include "camera_manager.h"
#include "camera_params.h"

class CameraControlPanel {
public:
    explicit CameraControlPanel(evt::CameraManager& camera_mgr);
    
    void render();
    
    // Camera selection and configuration
    void renderCameraList(const std::vector<GigEVisionDeviceInfo>& devices,
                         std::vector<bool>& selected);
    void renderCameraSettings();
    
    // Streaming controls                     
    void renderStreamingControls();
    
    // Recording controls
    void renderRecordingControls(const std::string& output_folder,
                                const std::string& encoder_setup);

private:
    evt::CameraManager& camera_manager;
    bool show_settings{false};
    bool show_temperature{false};
};