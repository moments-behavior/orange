#include "camera_control_panel.h"
#include <sstream>

CameraControlPanel::CameraControlPanel(evt::CameraManager& camera_mgr)
    : camera_manager(camera_mgr), device_info(nullptr), 
      selected_cameras(nullptr), known_cameras_(nullptr) {
    // Validate camera manager is properly initialized
    if (camera_mgr.getCameraCount() > 0) {
        LOG(INFO) << "CameraControlPanel initialized with " 
                  << camera_mgr.getCameraCount() << " cameras";
    }
}

void CameraControlPanel::render() {
    if (ImGui::CollapsingHeader("Available Cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderCameraList();
    }

    if (ImGui::CollapsingHeader("Camera Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderCameraSettings();
    }
    
    renderStreamingControls();
}

void CameraControlPanel::renderCameraList() {
    if (!device_info || !selected_cameras) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No cameras found");
        return;
    }

    if (device_info->empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No cameras found");
        return;
    }

    // Create a table to display camera information
    if (ImGui::BeginTable("CameraList", 4, 
        ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY | 
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders,
        ImVec2(0, 100))) {
        
        // Set fixed widths for each column
        ImGui::TableSetupColumn("Select", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Serial", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("IP Address", ImGuiTableColumnFlags_WidthFixed, 120.0f);

        // Set table height
        ImGui::TableSetupScrollFreeze(0, 1);  // Freeze header row

        for (size_t i = 0; i < device_info->size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            
            // Checkbox for camera selection
            bool was_selected = (*selected_cameras)[i];
            bool is_selected = (*selected_cameras)[i];
            if (ImGui::Checkbox(("##select" + std::to_string(i)).c_str(), &is_selected)) {
                (*selected_cameras)[i] = is_selected;
                if (is_selected && !was_selected) {
                    LOG(INFO) << "Selected camera " << i;
                } else if (!is_selected && was_selected) {
                    LOG(INFO) << "Deselected camera " << i;
                }
            }

            ImGui::TableNextColumn();
            ImGui::Text("%s", (*device_info)[i].userDefinedName);
            ImGui::TableNextColumn();
            ImGui::Text("%s", (*device_info)[i].serialNumber);
            ImGui::TableNextColumn();
            ImGui::Text("%s", (*device_info)[i].currentIp);
        }
        ImGui::EndTable();
    }
}

void CameraControlPanel::renderCameraSettings() {
    // Validate all required pointers
    if (!device_info || !selected_cameras || !known_cameras_) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
            "Camera info not initialized");
        return;
    }

    // Early validation of camera manager state
    if (camera_manager.getCameraCount() == 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
            "No cameras initialized");
        return;
    }

    static bool debug_rendering = false;  // Add debug toggle
    if (ImGui::Checkbox("Debug Rendering", &debug_rendering)) {
        LOG(INFO) << "Rendering debug " << (debug_rendering ? "enabled" : "disabled");
    }

    // Create tabs for each selected camera
    if (ImGui::BeginTabBar("CameraSettingsTabs")) {
        for (size_t i = 0; i < device_info->size(); i++) {
            if (!(*selected_cameras)[i]) continue;

            // Get camera serial and lookup config
            std::string serial((*device_info)[i].serialNumber);
            auto config_it = known_cameras_->find(serial);
            if (config_it == known_cameras_->end()) continue;
            
            if (debug_rendering) {
                LOG(INFO) << "Processing camera " << serial;
            }

            // Validate camera instance exists and is accessible
            const auto& camera_instance = camera_manager.getCamera(i);
            if (!camera_instance.camera) {
                ImGui::EndTabBar();
                return;
            }

            // Create tab label
            std::string label = config_it != known_cameras_->end() ? 
                              std::string("Cam") + std::to_string(i+1) : serial;
            
            if (ImGui::BeginTabItem(label.c_str())) {
                if (debug_rendering) {
                    LOG(INFO) << "Starting tab for camera " << serial;
                }

                bool camera_ready = camera_instance.camera && camera_instance.camera->isOpen();

                // Get actual ranges from camera instead of config
                evt::EmergentCamera::ParameterRange exposure_range = camera_manager.getExposureRange(i);
                evt::EmergentCamera::ParameterRange gain_range = camera_manager.getGainRange(i);
                evt::EmergentCamera::ParameterRange frame_rate_range = camera_manager.getFrameRateRange(i);
                evt::EmergentCamera::ParameterRange iris_range = camera_manager.getIrisRange(i);
                evt::EmergentCamera::ParameterRange focus_range = camera_manager.getFocusRange(i);
                
                // Camera Info section in a table
                if (ImGui::BeginTable("CameraInfo", 2, 
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    
                    ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                    ImGui::TableSetupColumn("Value");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Serial");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", serial.c_str());

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("IP Address");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", (*device_info)[i].currentIp);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("MAC Address");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", (*device_info)[i].macAddress);

                    ImGui::EndTable();
                }

                ImGui::Spacing();

                // Only show camera settings if camera is ready
                if (camera_ready && config_it != known_cameras_->end()) {
                    // Need to cast away const to modify the params
                    auto& params = const_cast<evt::CameraParams&>(config_it->second);

                    // Clamp all values to valid ranges before displaying
                    params.exposure = std::clamp(static_cast<int>(params.exposure), 
                        static_cast<int>(exposure_range.min), 
                        static_cast<int>(exposure_range.max));
                    params.gain = std::clamp(static_cast<int>(params.gain),
                        static_cast<int>(gain_range.min),
                        static_cast<int>(gain_range.max));
                    params.frame_rate = std::clamp(static_cast<int>(params.frame_rate),
                        static_cast<int>(frame_rate_range.min),
                        static_cast<int>(frame_rate_range.max));
                    params.iris = std::clamp(static_cast<int>(params.iris),
                        static_cast<int>(iris_range.min),
                        static_cast<int>(iris_range.max));
                    params.focus = std::clamp(static_cast<int>(params.focus),
                        static_cast<int>(focus_range.min),
                        static_cast<int>(focus_range.max));

                    // Get current camera state before rendering controls
                    auto current_state = camera_instance.camera->getCurrentState();
                    
                    // Update params with actual camera values
                    params.exposure = current_state.exposure;
                    params.gain = current_state.gain;
                    params.frame_rate = current_state.frame_rate;
                    params.iris = current_state.iris;
                    params.focus = current_state.focus;

                    if (ImGui::BeginTable("CameraSettings", 4,  // Changed from 3 to 4 columns 
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        
                        ImGui::TableSetupColumn("Setting", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                        ImGui::TableSetupColumn("Value");
                        ImGui::TableSetupColumn("Input", ImGuiTableColumnFlags_WidthFixed, 100.0f);  // New column
                        ImGui::TableSetupColumn("Range", ImGuiTableColumnFlags_WidthFixed, 150.0f);

                        // Exposure row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Exposure");
                        ImGui::TableNextColumn();
                        int exposure = current_state.exposure;
                        if (ImGui::SliderInt("##exposure_slider", &exposure, 
                            exposure_range.min, exposure_range.max)) 
                        {
                            updateCameraParameter(i, serial, "exposure", exposure, params);
                        }
                        ImGui::TableNextColumn();
                        if (ImGui::InputInt("##exposure_input", &exposure, 0)) {
                            exposure = std::clamp(exposure, 
                                static_cast<int>(exposure_range.min), 
                                static_cast<int>(exposure_range.max));
                            updateCameraParameter(i, serial, "exposure", exposure, params);
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d μs", exposure_range.min, exposure_range.max);

                        // Gain row with same pattern
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Gain");
                        ImGui::TableNextColumn();
                        int gain = current_state.gain;
                        if (ImGui::SliderInt("##gain_slider", &gain, 
                            gain_range.min, gain_range.max)) 
                        {
                            updateCameraParameter(i, serial, "gain", gain, params);
                        }
                        ImGui::TableNextColumn();
                        if (ImGui::InputInt("##gain_input", &gain, 0)) {
                            gain = std::clamp(gain, 
                                static_cast<int>(gain_range.min), 
                                static_cast<int>(gain_range.max));
                            updateCameraParameter(i, serial, "gain", gain, params);
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d", gain_range.min, gain_range.max);

                        // Frame rate row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Frame Rate");
                        ImGui::TableNextColumn();
                        int frame_rate = current_state.frame_rate;
                        if (ImGui::SliderInt("##framerate", &frame_rate,
                            frame_rate_range.min, frame_rate_range.max))
                        {
                            updateCameraParameter(i, serial, "frame_rate", frame_rate, params);
                        }
                        ImGui::TableNextColumn();
                        if (ImGui::InputInt("##framerate_input", &frame_rate, 0)) {
                            frame_rate = std::clamp(frame_rate, 
                                static_cast<int>(frame_rate_range.min), 
                                static_cast<int>(frame_rate_range.max));
                            updateCameraParameter(i, serial, "frame_rate", frame_rate, params);
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d fps", frame_rate_range.min, frame_rate_range.max);

                        // Iris row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Iris");
                        ImGui::TableNextColumn();
                        int iris = current_state.iris;
                        if (ImGui::SliderInt("##iris", &iris, 
                            iris_range.min, iris_range.max)) 
                        {
                            updateCameraParameter(i, serial, "iris", iris, params);
                        }
                        ImGui::TableNextColumn();
                        if (ImGui::InputInt("##iris_input", &iris, 0)) {
                            iris = std::clamp(iris, 
                                static_cast<int>(iris_range.min), 
                                static_cast<int>(iris_range.max));
                            updateCameraParameter(i, serial, "iris", iris, params);
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d", iris_range.min, iris_range.max);

                        // Focus row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Focus");
                        ImGui::TableNextColumn();
                        int focus = current_state.focus;
                        if (ImGui::SliderInt("##focus", &focus, 
                            focus_range.min, focus_range.max)) 
                        {
                            updateCameraParameter(i, serial, "focus", focus, params);
                        }
                        ImGui::TableNextColumn();
                        if (ImGui::InputInt("##focus_input", &focus, 0)) {
                            focus = std::clamp(focus, 
                                static_cast<int>(focus_range.min), 
                                static_cast<int>(focus_range.max));
                            updateCameraParameter(i, serial, "focus", focus, params);
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d", focus_range.min, focus_range.max);

                        ImGui::EndTable();
                    }
                }

                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
}

void CameraControlPanel::renderStreamingControls() {
    if (!camera_manager.getCameraCount()) {
        return;
    }

    for (size_t i = 0; i < camera_manager.getCameraCount(); i++) {
        const auto& camera = camera_manager.getCamera(i);
        std::string label = "Stream Camera " + std::to_string(i);
        
        bool is_streaming = camera_manager.isStreaming(i);
        if (ImGui::Checkbox(label.c_str(), &is_streaming)) {
            try {
                if (is_streaming) {
                    camera_manager.startStreaming(i, true); // Enable GPU
                } else {
                    camera_manager.stopStreaming(i);
                }
            } catch (const evt::CameraException& e) {
                // Handle error
            }
        }
    }
}

void CameraControlPanel::renderRecordingControls(
    const std::string& output_folder,
    const std::string& encoder_setup) {
    
    bool is_recording = camera_manager.isRecording();
    if (ImGui::Checkbox("Record", &is_recording)) {
        try {
            if (is_recording) {
                camera_manager.startRecording(output_folder, encoder_setup);
            } else {
                camera_manager.stopRecording();
            }
        } catch (const evt::CameraException& e) {
            // Handle error
        }
    }
}

void CameraControlPanel::updateCameraParameter(size_t camera_idx, const std::string& serial, 
                             const std::string& param_name, int value, 
                             evt::CameraParams& params) {
    try {
        LOG(INFO) << "Attempting to update " << param_name << " to " << value;
        camera_manager.getCamera(camera_idx).camera->logCurrentState("Before " + param_name + " change");
        
        if (param_name == "exposure") {
            camera_manager.updateExposure(camera_idx, value);
            params.exposure = value;
        } else if (param_name == "gain") {
            camera_manager.updateGain(camera_idx, value);
            params.gain = value;
        } else if (param_name == "frame_rate") {
            camera_manager.updateFrameRate(camera_idx, value);
            params.frame_rate = value;
        } else if (param_name == "iris") {
            camera_manager.updateIris(camera_idx, value);
            params.iris = value;
        } else if (param_name == "focus") {
            camera_manager.updateFocus(camera_idx, value);
            params.focus = value;
        }
        
        camera_manager.getCamera(camera_idx).camera->logCurrentState("After " + param_name + " change");
        LOG(INFO) << "Updated camera " << serial << " " << param_name << " to " << value;
    } catch (const evt::CameraException& e) {
        LOG(ERROR) << "Failed to update " << param_name << ": " << e.what();
    }
}