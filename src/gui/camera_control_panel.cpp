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
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
        
        ImGui::TableSetupColumn("Select", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Serial");
        ImGui::TableSetupColumn("IP Address");
        ImGui::TableHeadersRow();

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
        LOG(INFO) << "Early return: Missing required pointers";
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
            "Camera info not initialized");
        return;
    }

    // Early validation of camera manager state
    if (camera_manager.getCameraCount() == 0) {
        LOG(INFO) << "Early return: No cameras in manager";
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
            "No cameras initialized");
        return;
    }

    LOG(INFO) << "Starting to render camera settings tabs";
    
    // Create tabs for each selected camera
    if (ImGui::BeginTabBar("CameraSettingsTabs")) {
        LOG(INFO) << "Beginning tab bar";
        
        for (size_t i = 0; i < device_info->size(); i++) {
            LOG(INFO) << "Checking camera " << i << " (selected: " << (*selected_cameras)[i] << ")";
            
            if (!(*selected_cameras)[i]) {
                LOG(INFO) << "Camera " << i << " not selected, skipping";
                continue;
            }

            // Get camera serial and lookup config
            std::string serial((*device_info)[i].serialNumber);
            LOG(INFO) << "Processing camera " << serial;
            
            auto config_it = known_cameras_->find(serial);
            if (config_it == known_cameras_->end()) {
                LOG(INFO) << "No config found for camera " << serial << ", skipping";
                continue;
            }
            
            // Validate camera instance exists and is accessible
            const auto& camera_instance = camera_manager.getCamera(i);
            if (!camera_instance.camera) {
                LOG(INFO) << "Camera instance invalid for " << serial;
                ImGui::EndTabBar();
                return;
            }

            LOG(INFO) << "Starting tab for camera " << serial;

            // Create tab label
            std::string label = config_it != known_cameras_->end() ? 
                              std::string("Cam") + std::to_string(i+1) : serial;
            
            if (ImGui::BeginTabItem(label.c_str())) {
                // Initialize ranges with defaults
                evt::EmergentCamera::ParameterRange exposure_range{0, 1000000, 1};
                evt::EmergentCamera::ParameterRange gain_range{0, 100, 1};
                evt::EmergentCamera::ParameterRange frame_rate_range{1, 120, 1};
                evt::EmergentCamera::ParameterRange iris_range{0, 100, 1};
                evt::EmergentCamera::ParameterRange focus_range{0, 100, 1};

                // Try to get actual ranges from camera
                try {
                    LOG(INFO) << "Getting parameter ranges for camera " << serial;
                    
                    exposure_range = camera_manager.getExposureRange(i);
                    LOG(INFO) << "Exposure range: " << exposure_range.min 
                            << " - " << exposure_range.max 
                            << " (increment: " << exposure_range.increment << ")";
                    
                    gain_range = camera_manager.getGainRange(i);
                    LOG(INFO) << "Gain range: " << gain_range.min 
                            << " - " << gain_range.max
                            << " (increment: " << gain_range.increment << ")";
                    
                    frame_rate_range = camera_manager.getFrameRateRange(i);
                    LOG(INFO) << "Frame rate range: " << frame_rate_range.min 
                            << " - " << frame_rate_range.max
                            << " (increment: " << frame_rate_range.increment << ")";
                    
                    iris_range = camera_manager.getIrisRange(i);
                    LOG(INFO) << "Iris range: " << iris_range.min 
                            << " - " << iris_range.max
                            << " (increment: " << iris_range.increment << ")";
                    
                    focus_range = camera_manager.getFocusRange(i);
                    LOG(INFO) << "Focus range: " << focus_range.min 
                            << " - " << focus_range.max
                            << " (increment: " << focus_range.increment << ")";
                    
                } catch (const evt::CameraException& e) {
                    LOG(WARNING) << "Could not get camera ranges: " << e.what();
                    // Continue with default ranges
                }

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

                // Camera Settings in a table
                if (config_it != known_cameras_->end()) {
                    // Need to cast away const to modify the params
                    auto& params = const_cast<evt::CameraParams&>(config_it->second);
                    
                    if (ImGui::BeginTable("CameraSettings", 3, 
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        
                        ImGui::TableSetupColumn("Setting", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                        ImGui::TableSetupColumn("Value");
                        ImGui::TableSetupColumn("Range", ImGuiTableColumnFlags_WidthFixed, 150.0f);

                        // Exposure row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Exposure");
                        ImGui::TableNextColumn();
                        int exposure = params.exposure;
                        if (ImGui::SliderInt("##exposure", &exposure, 
                            exposure_range.min, exposure_range.max)) 
                        {
                            try {
                                LOG(INFO) << "Attempting to update exposure to " << exposure;
                                camera_manager.getCamera(i).camera->logCurrentState("Before exposure change");
                                
                                camera_manager.updateExposure(i, exposure);
                                params.exposure = exposure;
                                
                                camera_manager.getCamera(i).camera->logCurrentState("After exposure change");
                                LOG(INFO) << "Updated camera " << serial << " exposure to " << exposure;
                            } catch (const evt::CameraException& e) {
                                LOG(ERROR) << "Failed to update exposure: " << e.what();
                            }
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d μs", exposure_range.min, exposure_range.max);

                        // Gain row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Gain");
                        ImGui::TableNextColumn();
                        int gain = params.gain;
                        if (ImGui::SliderInt("##gain", &gain, 
                            gain_range.min, gain_range.max)) 
                        {
                            try {
                                LOG(INFO) << "Attempting to update gain to " << gain;
                                camera_manager.getCamera(i).camera->logCurrentState("Before gain change");
                                
                                camera_manager.updateGain(i, gain);
                                params.gain = gain;  // Update stored value
                                
                                camera_manager.getCamera(i).camera->logCurrentState("After gain change");
                                LOG(INFO) << "Updated camera " << serial << " gain to " << gain;
                            } catch (const evt::CameraException& e) {
                                LOG(ERROR) << "Failed to update gain: " << e.what();
                            }
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d", gain_range.min, gain_range.max);

                        // Frame rate row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Frame Rate");
                        ImGui::TableNextColumn();
                        int frame_rate = params.frame_rate;
                        if (ImGui::SliderInt("##framerate", &frame_rate,
                            frame_rate_range.min, frame_rate_range.max))
                        {
                            try {
                                LOG(INFO) << "Attempting to update frame rate to " << frame_rate;
                                camera_manager.getCamera(i).camera->logCurrentState("Before frame rate change");
                                
                                camera_manager.updateFrameRate(i, frame_rate);
                                params.frame_rate = frame_rate;  // Update stored value
                                
                                camera_manager.getCamera(i).camera->logCurrentState("After frame rate change");
                                LOG(INFO) << "Updated camera " << serial << " frame rate to " << frame_rate;
                            } catch (const evt::CameraException& e) {
                                LOG(ERROR) << "Failed to update frame rate: " << e.what();
                            }
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d fps", frame_rate_range.min, frame_rate_range.max);

                        // Iris row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Iris");
                        ImGui::TableNextColumn();
                        int iris = params.iris;
                        if (ImGui::SliderInt("##iris", &iris, 
                            iris_range.min, iris_range.max)) 
                        {
                            try {
                                LOG(INFO) << "Attempting to update iris to " << iris;
                                camera_manager.getCamera(i).camera->logCurrentState("Before iris change");
                                
                                camera_manager.updateIris(i, iris);
                                params.iris = iris;  // Update stored value
                                
                                camera_manager.getCamera(i).camera->logCurrentState("After iris change");
                                LOG(INFO) << "Updated camera " << serial << " iris to " << iris;
                            } catch (const evt::CameraException& e) {
                                LOG(ERROR) << "Failed to update iris: " << e.what();
                            }
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%d - %d", iris_range.min, iris_range.max);

                        // Focus row
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Focus");
                        ImGui::TableNextColumn();
                        int focus = params.focus;
                        if (ImGui::SliderInt("##focus", &focus, 
                            focus_range.min, focus_range.max)) 
                        {
                            try {
                                LOG(INFO) << "Attempting to update focus to " << focus;
                                camera_manager.getCamera(i).camera->logCurrentState("Before focus change");
                                
                                camera_manager.updateFocus(i, focus);
                                params.focus = focus;  // Update stored value
                                
                                camera_manager.getCamera(i).camera->logCurrentState("After focus change");
                                LOG(INFO) << "Updated camera " << serial << " focus to " << focus;
                            } catch (const evt::CameraException& e) {
                                LOG(ERROR) << "Failed to update focus: " << e.what();
                            }
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