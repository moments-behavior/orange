#include "main_window.h"
#include "imgui.h"
#include "implot.h"
#include "camera_manager.h"
#include "emergent_camera.h"
#include "camera_control_panel.h"  // Add this include
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
                auto config = loadCameraConfig(entry.path().string());
                // Store config using MAC address as key
                if (!config.params.camera_serial.empty()) {
                    known_cameras_[config.params.camera_serial] = config;
                    LOG(INFO) << "Loaded config for camera: " << config.params.camera_name;
                }
            }
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error loading camera configs: " << e.what();
    }
}

CameraConfig MainWindow::loadCameraConfig(const std::string& config_file) {
    CameraConfig config;
    try {
        std::ifstream f(config_file);
        nlohmann::json j;
        f >> j;
        
        config.params = evt::CameraParams::from_json(j);
        config.name = j.value("name", "");
        config.nic_port = j.value("nic_port", "");
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error parsing config file " << config_file << ": " << e.what();
    }
    return config;
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
        
        window_ctx_ = std::make_unique<gx_context>();
        if (!window_ctx_) {
            LOG(ERROR) << "Failed to create window context";
            throw std::runtime_error("Window context creation failed");
        }
        
        // Initialize the context fields
        window_ctx_->width = 1920;
        window_ctx_->height = 1080;
        window_ctx_->render_target = nullptr;
        window_ctx_->swap_interval = 1;
        strncpy(window_ctx_->glsl_version, "#version 130", sizeof(window_ctx_->glsl_version) - 1);
        window_ctx_->glsl_version[sizeof(window_ctx_->glsl_version) - 1] = '\0';
        
        LOG(INFO) << "Window context created successfully";
        
        camera_manager_ = std::make_unique<evt::CameraManager>();
        if (!camera_manager_) {
            LOG(ERROR) << "Failed to create camera manager";
            throw std::runtime_error("Camera manager creation failed");
        }
        LOG(INFO) << "Camera manager created successfully";
        
        // Create camera control panel after camera manager is initialized
        camera_control_panel_ = std::make_unique<CameraControlPanel>(*camera_manager_);
        if (!camera_control_panel_) {
            LOG(ERROR) << "Failed to create camera control panel";
            throw std::runtime_error("Camera control panel creation failed");
        }
        LOG(INFO) << "Camera control panel created successfully";

        // Initialize default encoder config
        encoder_config_.encoder_codec = "h264";
        encoder_config_.encoder_preset = "p1";
        encoder_config_.UpdateEncoderSetup();
        
        // Update recording folder path to use home directory
        recording_folder_ = (fs_utils::get_home_directory() / "orange_data" / "recordings").string();
        
        // Update config loading path to use home directory
        loadCameraConfigs((fs_utils::get_home_directory() / "orange_data" / "config/local").string());
        
        LOG(INFO) << "MainWindow construction completed successfully";
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during MainWindow construction: " << e.what();
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

void MainWindow::initializeCameras() {
    LOG(INFO) << "Scanning for cameras...";
    
    try {
        // Clear existing device info
        device_info_.clear();
        selected_cameras_.clear();

        // Start with a reasonable buffer size
        unsigned int buffer_size = 16;  // Should be enough for most cases
        unsigned int actual_num = 0;
        std::vector<GigEVisionDeviceInfo> temp_devices(buffer_size);

        // List all connected devices
        EVT_ERROR err = Emergent::EVT_ListDevices(temp_devices.data(), &buffer_size, &actual_num);
        if (err != EVT_SUCCESS) {
            LOG(ERROR) << "Failed to list cameras: " << err;
            return;
        }

        LOG(INFO) << "Found " << actual_num << " cameras";
        
        // Resize our vector to match actual number of devices found
        device_info_.resize(actual_num);
        selected_cameras_.resize(actual_num, false);  // Initialize all as unselected

        // Copy over the device info
        for (size_t i = 0; i < actual_num; i++) {
            device_info_[i] = temp_devices[i];
            
            // Try to match against known cameras from config
            bool found_in_config = false;
            std::string mac_addr(device_info_[i].macAddress);
            for (const auto& [config_mac, camera_info] : known_cameras_) {
                if (mac_addr == config_mac) {
                    found_in_config = true;
                    LOG(INFO) << "Found configured camera: " 
                             << "\n  Name: " << device_info_[i].userDefinedName
                             << "\n  Serial: " << device_info_[i].serialNumber
                             << "\n  IP: " << device_info_[i].currentIp
                             << "\n  MAC: " << device_info_[i].macAddress
                             << "\n  NIC Port: " << camera_info.nic_port;
                    break;
                }
            }

            if (!found_in_config) {
                LOG(WARNING) << "Found unconfigured camera: "
                            << "\n  Name: " << device_info_[i].userDefinedName
                            << "\n  Serial: " << device_info_[i].serialNumber
                            << "\n  IP: " << device_info_[i].currentIp
                            << "\n  MAC: " << device_info_[i].macAddress;
            }
        }

        // Initialize the camera manager with found devices
        if (!device_info_.empty()) {
            std::vector<std::string> config_files(device_info_.size());
            camera_manager_->initializeCameras(selected_cameras_, device_info_, config_files);
            LOG(INFO) << "Camera manager initialized with " << device_info_.size() << " cameras";
        }

        // After initializing device_info_ and selected_cameras_, update the control panel
        camera_control_panel_->setDeviceInfo(device_info_);
        camera_control_panel_->setSelectedCameras(selected_cameras_);
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during camera initialization: " << e.what();
    }
}

void MainWindow::render() {
    create_new_frame();

    // Calculate status bar height for proper layout
    float status_bar_height = ImGui::GetFrameHeight();
    
    // Create main window that takes up full screen minus status bar height
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 
        ImGui::GetIO().DisplaySize.y - status_bar_height));
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
    camera_control_panel_->render();
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
        initializeWindow();
        LOG(INFO) << "Window initialization completed";
        
        setupStyle();
        LOG(INFO) << "Style setup completed";
        
        // Initialize cameras
        initializeCameras();

        LOG(INFO) << "MainWindow initialization completed successfully";
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during MainWindow initialization: " << e.what();
        throw; // Re-throw to ensure proper cleanup
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

void MainWindow::cleanup() {
    if (camera_manager_) {
        // Stop all cameras
        for (size_t i = 0; i < camera_manager_->getCameraCount(); i++) {
            if (camera_manager_->isStreaming(i)) {
                camera_manager_->stopStreaming(i);
            }
        }
    }
    
    if (window_ctx_) {
        gx_cleanup(window_ctx_.get());
    }
}