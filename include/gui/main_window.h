#pragma once

#include "camera_manager.h"
#include "gx_helper.h"
#include "encoder_config.h"
#include <memory>
#include <vector>

class MainWindow {
public:
    MainWindow();
    ~MainWindow();

    void initialize();
    void render();
    bool shouldClose() const;
    void cleanup();

private:
    void renderMainMenu();
    void renderCameraControls();
    void renderStatusBar();
    void initializeWindow();
    void setupStyle();

    // Window context
    std::unique_ptr<gx_context> window_ctx_;
    bool show_demo_window_ = false;
    
    // Camera management
    std::unique_ptr<evt::CameraManager> camera_manager_;
    std::vector<bool> selected_cameras_;
    std::vector<GigEVisionDeviceInfo> device_info_;
    
    // Configuration
    EncoderConfig encoder_config_;
    std::string recording_folder_;
    bool is_recording_ = false;

    // GUI state
    bool show_camera_properties_ = false;
    bool show_encoder_settings_ = false;
};
