#pragma once

// First include all third-party and standard headers
#include <memory>
#include <vector>
#include <string>

// Then include our core dependencies
#include "camera_manager.h"
#include "gx_helper.h"
#include "encoder_config.h"
#include "json.hpp"
#include "camera_params.h"

// Finally include our complete class definitions
#include "camera_preview_window.h"  // Must come before camera_control_panel.h
#include "camera_control_panel.h"

class MainWindow {
public:
    // Constructor & Destructor
    MainWindow();
    ~MainWindow();

    // Delete copy constructor and assignment operator
    MainWindow(const MainWindow&) = delete;
    MainWindow& operator=(const MainWindow&) = delete;

    // Core window functions
    void initialize();
    void render();
    bool shouldClose() const;
    void cleanup();

private:
    // Window initialization and setup
    void initializeWindow();
    void setupStyle();
    void initializeCameras();

    // Rendering functions
    void renderMainMenu();
    void renderStatusBar();

    // Modify method signature
    void loadCameraConfigs(const std::string& config_path); // Move to private

    // Window context and graphics
    std::unique_ptr<gx_context> window_ctx_;
    bool show_demo_window_ = false;
    
    // Camera management
    std::unique_ptr<evt::CameraManager> camera_manager_;
    std::vector<bool> selected_cameras_;
    std::vector<GigEVisionDeviceInfo> device_info_;
    std::unordered_map<std::string, evt::CameraParams> known_cameras_;
    
    // Add CameraControlPanel
    std::unique_ptr<CameraControlPanel> camera_control_panel_;
    
    // Add preview window member
    std::unique_ptr<CameraPreviewWindow> preview_window_;  // Now CameraPreviewWindow is fully defined

    // Configuration and settings
    EncoderConfig encoder_config_;
    std::string recording_folder_;
    bool is_recording_ = false;

    // GUI state
    bool show_camera_properties_ = false;
    bool show_encoder_settings_ = false;

    // Window dimensions - could be useful for responsive layout
    static constexpr int DEFAULT_WINDOW_WIDTH = 1920;
    static constexpr int DEFAULT_WINDOW_HEIGHT = 1080;
};