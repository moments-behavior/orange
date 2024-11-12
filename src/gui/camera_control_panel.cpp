#include "camera_control_panel.h"
#include <sstream>

CameraControlPanel::CameraControlPanel(evt::CameraManager& camera_mgr)
    : camera_manager(camera_mgr) {}

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

    // Add a fixed height container for the camera list
    float table_height = ImGui::GetTextLineHeightWithSpacing() * 8.0f; // Height for ~7 rows + header
    if (ImGui::BeginChild("CameraListContainer", ImVec2(0, table_height), true)) {
        // Create a table to display camera information
        if (ImGui::BeginTable("CameraList", 4, 
            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY | 
            ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
            
            // Make columns more proportional
            ImGui::TableSetupColumn("Select", ImGuiTableColumnFlags_WidthFixed, 50.0f);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Serial", ImGuiTableColumnFlags_WidthFixed, 120.0f);
            ImGui::TableSetupColumn("IP Address", ImGuiTableColumnFlags_WidthFixed, 120.0f);
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
                        try {
                            camera_manager.startStreaming(i);
                        } catch (const evt::CameraException& e) {
                            LOG(ERROR) << "Failed to start camera " << i << ": " << e.what();
                            (*selected_cameras)[i] = false;
                        }
                    } else if (!is_selected && was_selected) {
                        try {
                            camera_manager.stopStreaming(i);
                        } catch (const evt::CameraException& e) {
                            LOG(ERROR) << "Failed to stop camera " << i << ": " << e.what();
                        }
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
        ImGui::EndChild();
    }

    // Add some spacing after the table
    ImGui::Spacing();
}

void CameraControlPanel::renderCameraSettings() {
    if (!device_info || !selected_cameras) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
            "No cameras available");
        return;
    }

    if (camera_manager.getCameraCount() == 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
            "Select a camera to configure settings");
        return;
    }

    // Create tabs for each selected camera
    if (ImGui::BeginTabBar("CameraSettingsTabs")) {
        for (size_t i = 0; i < device_info->size(); i++) {
            if (!(*selected_cameras)[i]) continue;

            std::string label = std::string((*device_info)[i].userDefinedName) + 
                " (" + (*device_info)[i].serialNumber + ")";
            
            if (ImGui::BeginTabItem(label.c_str())) {
                // Create scrollable area for settings
                ImGui::BeginChild("Settings", ImVec2(0, 0), false);

                const auto& camera = camera_manager.getCamera(i);
                
                // Camera Info section
                if (ImGui::TreeNode("Camera Info")) {
                    ImGui::Text("Serial: %s", (*device_info)[i].serialNumber);
                    ImGui::Text("IP: %s", (*device_info)[i].currentIp);
                    ImGui::Text("MAC: %s", (*device_info)[i].macAddress);
                    ImGui::TreePop();
                }

                // Exposure control
                if (ImGui::TreeNode("Exposure Settings")) {
                    int exposure = camera.params.exposure;
                    if (ImGui::SliderInt("Exposure (μs)", &exposure, 
                        camera.params.exposure_min, 
                        camera.params.exposure_max)) 
                    {
                        try {
                            camera_manager.updateExposure(i, exposure);
                        } catch (const evt::CameraException& e) {
                            LOG(ERROR) << "Failed to update exposure: " << e.what();
                        }
                    }
                    ImGui::TreePop();
                }
                
                // Gain control
                if (ImGui::TreeNode("Gain Settings")) {
                    int gain = camera.params.gain;
                    if (ImGui::SliderInt("Gain", &gain, 
                        camera.params.gain_min, 
                        camera.params.gain_max)) 
                    {
                        try {
                            camera_manager.updateGain(i, gain);
                        } catch (const evt::CameraException& e) {
                            LOG(ERROR) << "Failed to update gain: " << e.what();
                        }
                    }
                    ImGui::TreePop();
                }
                
                // Frame rate control
                if (ImGui::TreeNode("Frame Rate Settings")) {
                    int frame_rate = camera.params.frame_rate;
                    if (ImGui::SliderInt("Frame Rate (fps)", &frame_rate,
                        camera.params.frame_rate_min,
                        camera.params.frame_rate_max))
                    {
                        try {
                            camera_manager.updateFrameRate(i, frame_rate);
                        } catch (const evt::CameraException& e) {
                            LOG(ERROR) << "Failed to update frame rate: " << e.what();
                        }
                    }
                    ImGui::TreePop();
                }

                // Resolution info
                if (ImGui::TreeNode("Resolution Settings")) {
                    ImGui::Text("Current: %dx%d", camera.params.width, camera.params.height);
                    ImGui::Text("Range: %d-%d x %d-%d", 
                        camera.params.width_min, camera.params.width_max,
                        camera.params.height_min, camera.params.height_max);
                    ImGui::TreePop();
                }

                // Temperature info if enabled
                if (show_temperature && ImGui::TreeNode("Temperature")) {
                    ImGui::Text("Sensor: %d°C", camera.params.sens_temp);
                    ImGui::Text("Range: %d°C - %d°C", 
                        camera.params.sens_temp_min,
                        camera.params.sens_temp_max);
                    ImGui::TreePop();
                }

                ImGui::EndChild();
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