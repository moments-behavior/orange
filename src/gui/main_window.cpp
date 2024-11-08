#include "main_window.h"
#include "imgui.h"
#include "implot.h"
#include <filesystem>
#include <chrono>

MainWindow::MainWindow() 
    : window_ctx_(std::make_unique<gx_context>()),
      camera_manager_(std::make_unique<evt::CameraManager>()) {
    
    // Initialize default encoder config
    encoder_config_.encoder_codec = "h264";
    encoder_config_.encoder_preset = "p7";
    encoder_config_.UpdateEncoderSetup();
    
    // Set default recording folder
    recording_folder_ = "recordings";
}

MainWindow::~MainWindow() {
    cleanup();
}

void MainWindow::initialize() {
    initializeWindow();
    setupStyle();

    // Initialize cameras
    // TODO: Scan for cameras and populate device_info_
}

void MainWindow::initializeWindow() {
    window_ctx_->width = 1920;
    window_ctx_->height = 1080;
    window_ctx_->render_target = gx_glfw_init_render_target(3, 3, 
        window_ctx_->width, window_ctx_->height, 
        "Camera Control", window_ctx_->glsl_version);
    
    gx_init(window_ctx_.get(), window_ctx_->render_target);
    gx_imgui_init(window_ctx_.get());
}

void MainWindow::setupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    
    // Set a dark color scheme
    ImGui::StyleColorsDark();
}

void MainWindow::render() {
    create_new_frame();
    
    // Create main window that takes up full screen
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
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
    ImGui::BeginChild("Camera Feeds");
    // TODO: Render camera preview windows
    ImGui::EndChild();
    
    ImGui::NextColumn();
    
    // Right column: Controls
    ImGui::BeginChild("Controls");
    renderCameraControls();
    ImGui::EndChild();
    
    ImGui::Columns(1);
    
    renderStatusBar();
    
    ImGui::End();

    // Demo window if enabled
    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }

    render_a_frame(window_ctx_.get());
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

void MainWindow::renderCameraControls() {
    // Camera selection
    if (ImGui::CollapsingHeader("Cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < device_info_.size(); i++) {
            if (ImGui::Selectable(device_info_[i].userDefinedName, selected_cameras_[i])) {
                selected_cameras_[i] = !selected_cameras_[i];
                
                if (selected_cameras_[i]) {
                    camera_manager_->startStreaming(i);
                } else {
                    camera_manager_->stopStreaming(i);
                }
            }
        }
    }

    // Camera properties window
    if (show_camera_properties_) {
        ImGui::Begin("Camera Properties", &show_camera_properties_);
        
        if (ImGui::BeginTabBar("CameraSettings")) {
            for (size_t i = 0; i < camera_manager_->getCameraCount(); i++) {
                if (ImGui::BeginTabItem(std::to_string(i).c_str())) {
                    auto& camera = camera_manager_->getCamera(i);
                    
                    // Exposure control
                    int exposure = camera.params.exposure;
                    if (ImGui::SliderInt("Exposure", &exposure, 0, 10000)) {
                        camera_manager_->updateExposure(i, exposure);
                    }
                    
                    // Gain control
                    int gain = camera.params.gain;
                    if (ImGui::SliderInt("Gain", &gain, 0, 100)) {
                        camera_manager_->updateGain(i, gain);
                    }
                    
                    // Frame rate control
                    int frame_rate = camera.params.frame_rate;
                    if (ImGui::SliderInt("Frame Rate", &frame_rate, 1, 120)) {
                        camera_manager_->updateFrameRate(i, frame_rate);
                    }
                    
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

    // Encoder settings window
    if (show_encoder_settings_) {
        ImGui::Begin("Encoder Settings", &show_encoder_settings_);
        
        const char* codecs[] = { "h264", "hevc" };
        int current_codec = 0;
        for (int i = 0; i < IM_ARRAYSIZE(codecs); i++) {
            if (encoder_config_.encoder_codec == codecs[i]) {
                current_codec = i;
                break;
            }
        }
        
        if (ImGui::Combo("Codec", &current_codec, codecs, IM_ARRAYSIZE(codecs))) {
            encoder_config_.encoder_codec = codecs[current_codec];
            encoder_config_.UpdateEncoderSetup();
        }
        
        const char* presets[] = { "p1", "p3", "p5", "p7" };
        int current_preset = 0;
        for (int i = 0; i < IM_ARRAYSIZE(presets); i++) {
            if (encoder_config_.encoder_preset == presets[i]) {
                current_preset = i;
                break;
            }
        }
        
        if (ImGui::Combo("Preset", &current_preset, presets, IM_ARRAYSIZE(presets))) {
            encoder_config_.encoder_preset = presets[current_preset];
            encoder_config_.UpdateEncoderSetup();
        }
        
        ImGui::End();
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