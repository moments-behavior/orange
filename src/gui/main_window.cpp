#include "main_window.h"
#include "imgui.h"
#include "implot.h"
#include "camera_manager.h"
#include "emergent_camera.h"
#include "camera_control_panel.h"
#include "fs_utils.hpp"
#include <filesystem>
#include <chrono>
#include <fstream>

void MainWindow::loadCameraConfigs(const std::string& config_path) {
    LOG(INFO) << "Loading camera configs from: " << config_path;
    
    try {
        namespace fs = std::filesystem;
        for (const auto& entry : fs::directory_iterator(config_path)) {
            if (entry.path().extension() == ".json") {
                LOG(INFO) << "Found JSON file: " << entry.path().string();
                std::ifstream f(entry.path().string());
                nlohmann::json j;
                f >> j;
                
                auto params = evt::CameraParams::from_json(j);
                if (!params.camera_serial.empty()) {
                    LOG(INFO) << "Loading config with serial: " << params.camera_serial;
                    known_cameras_[params.camera_serial] = params;
                    LOG(INFO) << "Loaded config for camera: " << params.camera_serial;
                }
            }
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error loading camera configs: " << e.what();
    }
}

MainWindow::MainWindow() {
    LOG(INFO) << "Starting MainWindow construction...";
    
    try {
        // Initialize required directories first
        if (!fs_utils::initialize_directories()) {
            LOG(ERROR) << "Failed to initialize required directories";
            throw std::runtime_error("Directory initialization failed");
        }
        LOG(INFO) << "Directories initialized successfully";
        
        // Create window context
        window_ctx_ = std::make_unique<gx_context>();
        if (!window_ctx_) {
            LOG(ERROR) << "Failed to create window context";
            throw std::runtime_error("Window context creation failed");
        }
        
        // Initialize the context fields
        window_ctx_->width = DEFAULT_WINDOW_WIDTH;
        window_ctx_->height = DEFAULT_WINDOW_HEIGHT;
        window_ctx_->render_target = nullptr;
        window_ctx_->swap_interval = 1;
        strncpy(window_ctx_->glsl_version, "#version 130", sizeof(window_ctx_->glsl_version) - 1);
        window_ctx_->glsl_version[sizeof(window_ctx_->glsl_version) - 1] = '\0';
        
        LOG(INFO) << "Window context created successfully";
        
        // Create camera manager with explicit success check
        camera_manager_ = std::make_unique<evt::CameraManager>();
        if (!camera_manager_) {
            throw std::runtime_error("Failed to create camera manager");
        }
        LOG(INFO) << "Camera manager created successfully";
        
        // Load camera configs 
        loadCameraConfigs((fs_utils::get_home_directory() / "orange_data" / "config/local").string());
        LOG(INFO) << "Loaded " << known_cameras_.size() << " camera configurations";

        // Initialize encoder config with safe defaults
        encoder_config_.encoder_codec = "h264";
        encoder_config_.encoder_preset = "p1";
        encoder_config_.UpdateEncoderSetup();
        
        // Update recording folder path with explicit directory verification
        recording_folder_ = (fs_utils::get_home_directory() / "orange_data" / "recordings").string();
        if (!std::filesystem::exists(recording_folder_)) {
            std::filesystem::create_directories(recording_folder_);
        }
        
        LOG(INFO) << "MainWindow construction completed successfully";
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during MainWindow construction: " << e.what();
        cleanup(); // Ensure proper cleanup on failure
        throw;
    }
}

MainWindow::~MainWindow() {
    cleanup();
}

void MainWindow::initializeWindow() {
    LOG(INFO) << "Starting window initialization...";
    
    try {
        if (!window_ctx_) {
            LOG(ERROR) << "Window context is null during initialization";
            throw std::runtime_error("Null window context");
        }
        
        window_ctx_->render_target = gx_glfw_init_render_target(3, 3, 
            window_ctx_->width, window_ctx_->height, 
            "Orange", window_ctx_->glsl_version);
            
        if (!window_ctx_->render_target) {
            LOG(ERROR) << "Failed to create GLFW render target";
            throw std::runtime_error("Render target creation failed");
        }
        
        LOG(INFO) << "GLFW render target created successfully";
        
        if (!gx_init(window_ctx_.get(), window_ctx_->render_target)) {
            LOG(ERROR) << "Failed to initialize graphics context";
            throw std::runtime_error("Graphics initialization failed");
        }
        
        LOG(INFO) << "Graphics context initialized successfully";
        
        if (!gx_imgui_init(window_ctx_.get())) {
            LOG(ERROR) << "Failed to initialize ImGui";
            throw std::runtime_error("ImGui initialization failed");
        }
        
        LOG(INFO) << "ImGui initialized successfully";
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during window initialization: " << e.what();
        throw;
    }
}

void MainWindow::cleanup() {
    // First stop all cameras and release camera resources
    if (camera_manager_) {
        // Stop all cameras
        for (size_t i = 0; i < camera_manager_->getCameraCount(); i++) {
            try {
                if (camera_manager_->isStreaming(i)) {
                    camera_manager_->stopStreaming(i);
                }
            } catch (const std::exception& e) {
                LOG(ERROR) << "Error stopping camera " << i << ": " << e.what();
            }
        }
    }

    // Release camera manager and control panel before ImGui cleanup
    camera_control_panel_.reset();
    camera_manager_.reset();
    
    if (window_ctx_) {
        // Cleanup order matters - ImGui should be cleaned up before GLFW
        if (window_ctx_->render_target) {
            // Cleanup ImGui OpenGL3 Renderer
            ImGui_ImplOpenGL3_Shutdown();
            // Cleanup ImGui GLFW Integration
            ImGui_ImplGlfw_Shutdown();
            // Cleanup ImGui Context
            ImGui::DestroyContext();
            
            // Now cleanup GLFW
            glfwDestroyWindow(window_ctx_->render_target);
            window_ctx_->render_target = nullptr;
            
            // Terminate GLFW
            glfwTerminate();
        }
    }
    
    LOG(INFO) << "Cleanup completed successfully";
}

void MainWindow::initializeCameras() {
    LOG(INFO) << "Scanning for cameras...";
    
    try {
        // Clear existing device info
        device_info_.clear();
        selected_cameras_.clear();

        // List all connected devices
        unsigned int buffer_size = 16;
        unsigned int actual_num = 0;
        std::vector<GigEVisionDeviceInfo> temp_devices(buffer_size);

        EVT_ERROR err = Emergent::EVT_ListDevices(temp_devices.data(), &buffer_size, &actual_num);
        if (err != EVT_SUCCESS) {
            LOG(ERROR) << "Failed to list cameras: " << err;
            return;
        }

        LOG(INFO) << "Found " << actual_num << " cameras";
        
        // Resize our vectors
        device_info_.resize(actual_num);
        selected_cameras_.resize(actual_num, true);  // Initially select all cameras

        // Copy device info and initialize cameras
        for (size_t i = 0; i < actual_num; i++) {
            device_info_[i] = temp_devices[i];
            
            // Check if we have a config for this camera
            std::string serial(device_info_[i].serialNumber);
            LOG(INFO) << "Checking camera serial '" << serial << "' against known configs";
            
            if (known_cameras_.find(serial) != known_cameras_.end()) {
                LOG(INFO) << "Found configured camera: \n"
                         << "  Name: " << device_info_[i].userDefinedName << "\n"
                         << "  Serial: " << serial << "\n"
                         << "  IP: " << device_info_[i].currentIp;
                selected_cameras_[i] = true;  // Ensure camera is selected
            }
        }

        // Initialize camera manager with found devices
        if (!device_info_.empty()) {
            std::vector<std::string> config_files(device_info_.size());
            camera_manager_->initializeCameras(selected_cameras_, device_info_, config_files, known_cameras_);
            LOG(INFO) << "Camera manager initialized with " << camera_manager_->getCameraCount() << " active cameras";
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during camera initialization: " << e.what();
    }
}

void MainWindow::render() {
    create_new_frame();
    
    // Create main window that takes up full screen minus status bar height
    float menu_bar_height = ImGui::GetFrameHeight();
    float status_bar_height = ImGui::GetFrameHeight();
    
    ImGui::SetNextWindowPos(ImVec2(0, menu_bar_height));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 
        ImGui::GetIO().DisplaySize.y - menu_bar_height - status_bar_height));

    ImGui::Begin("Main Window", nullptr, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus);

    renderMainMenu();
    
    // Split view for camera feeds and controls
    ImGui::Columns(2);
    
    // Left column: Camera feeds
    ImGui::BeginChild("Camera Feeds", ImVec2(0, -5)); // Small padding at bottom
    // TODO: Render camera preview windows
    ImGui::EndChild();
    
    ImGui::NextColumn();
    
    // Right column: Controls using CameraControlPanel
    ImGui::BeginChild("Controls", ImVec2(0, -5)); // Small padding at bottom
    if (camera_control_panel_ && camera_control_panel_->isInitialized()) {
        camera_control_panel_->render();
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 
            "Camera control panel not initialized");
    }
    ImGui::EndChild();
    
    ImGui::Columns(1);
    
    ImGui::End();

    // Status bar at the bottom
    renderStatusBar();

    // Demo window if enabled
    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }

    render_a_frame(window_ctx_.get());
}

void MainWindow::initialize() {
    LOG(INFO) << "Starting MainWindow initialization...";
    
    try {
        // First initialize window and graphics
        initializeWindow();
        setupStyle();
        
        // Load camera configurations first
        loadCameraConfigs((fs_utils::get_home_directory() / "orange_data" / "config/local").string());
        LOG(INFO) << "Loaded " << known_cameras_.size() << " camera configurations";
        
        // Initialize cameras with configs
        initializeCameras();
        LOG(INFO) << "Initialized " << device_info_.size() << " cameras";
        
        // Create and setup control panel last
        camera_control_panel_ = std::make_unique<CameraControlPanel>(*camera_manager_);
        if (!camera_control_panel_) {
            throw std::runtime_error("Failed to create camera control panel");
        }

        // Ensure device info and camera configs are properly set
        if (!device_info_.empty()) {
            LOG(INFO) << "Setting up control panel with " << device_info_.size() << " devices";
            camera_control_panel_->setDeviceInfo(device_info_);
            camera_control_panel_->setSelectedCameras(selected_cameras_);
            camera_control_panel_->setKnownCameras(known_cameras_);
            
            if (!camera_control_panel_->isInitialized()) {
                LOG(ERROR) << "Control panel failed to initialize properly";
                throw std::runtime_error("Control panel initialization failed");
            }
            
            LOG(INFO) << "Camera control panel initialized successfully";
        } else {
            LOG(WARNING) << "No cameras found - control panel will be limited";
        }

        LOG(INFO) << "MainWindow initialization completed successfully";
    } catch (const std::exception& e) {
        LOG(ERROR) << "MainWindow initialization failed: " << e.what();
        throw;
    }
}

void MainWindow::setupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    
    // Set a dark color scheme
    ImGui::StyleColorsDark();
}

void MainWindow::renderMainMenu() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit")) {
                glfwSetWindowShouldClose(window_ctx_->render_target, true);
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Camera Properties", nullptr, &show_camera_properties_);
            ImGui::MenuItem("Encoder Settings", nullptr, &show_encoder_settings_);
            ImGui::MenuItem("Demo Window", nullptr, &show_demo_window_);
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Recording")) {
            if (ImGui::MenuItem("Start Recording", nullptr, false, !is_recording_)) {
                camera_manager_->startRecording(recording_folder_, encoder_config_.encoder_setup);
                is_recording_ = true;
            }
            if (ImGui::MenuItem("Stop Recording", nullptr, false, is_recording_)) {
                camera_manager_->stopRecording();
                is_recording_ = false;
            }
            ImGui::EndMenu();
        }
        
        ImGui::EndMainMenuBar();
    }
}

void MainWindow::renderStatusBar() {
    // Create a status bar using a window at the bottom of the screen
    ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoTitleBar;

    // Calculate status bar position and size
    float height = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetIO().DisplaySize.y - height));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, height));

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("StatusBar", nullptr, window_flags);
    
    // Show recording status
    if (is_recording_) {
        ImGui::Text("Recording... ");
        
        // Show recording duration
        static auto recording_start = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - recording_start);
        ImGui::SameLine();
        ImGui::Text("Duration: %02d:%02d:%02d", 
            static_cast<int>(duration.count()) / 3600,
            (static_cast<int>(duration.count()) % 3600) / 60,
            static_cast<int>(duration.count()) % 60);
    }
    
    // Show camera status
    ImGui::SameLine();
    ImGui::Text("Cameras: %zu/%zu", 
        camera_manager_->getCameraCount(),
        device_info_.size());

    ImGui::End();
    ImGui::PopStyleVar();
}

bool MainWindow::shouldClose() const {
    return glfwWindowShouldClose(window_ctx_->render_target);
}
