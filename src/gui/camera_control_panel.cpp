#include "camera_control_panel.h"
#include <sstream>

CameraControlPanel::CameraControlPanel(evt::CameraManager& camera_mgr)
    : camera_manager(camera_mgr) {}

void CameraControlPanel::render() {
    if (!ImGui::Begin("Camera Control")) {
        ImGui::End();
        return;
    }

    // Camera settings button
    if (ImGui::Button("Camera Settings")) {
        show_settings = !show_settings;
    }
    ImGui::SameLine();
    if (ImGui::Button("Show Temperature")) {
        show_temperature = !show_temperature;
    }
    
    // Render sub-panels based on state
    if (show_settings) {
        renderCameraSettings();
    }
    
    renderStreamingControls();
    
    ImGui::End();
}

void CameraControlPanel::renderCameraList(
    const std::vector<GigEVisionDeviceInfo>& devices,
    std::vector<bool>& selected) {
    
    if (ImGui::BeginTable("Cameras", 3, 
        ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders)) {
        
        ImGui::TableSetupColumn("ID");
        ImGui::TableSetupColumn("Serial");
        ImGui::TableSetupColumn("IP");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < devices.size(); i++) {
            ImGui::TableNextRow();
            
            ImGui::TableSetColumnIndex(0);
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Selectable(std::to_string(i).c_str(), selected[i],
                ImGuiSelectableFlags_None))
            {
                selected[i] = !selected[i]; // Toggle the selection manually
            }
            ImGui::PopID();
            
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s", devices[i].serialNumber);
            
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", devices[i].currentIp);
        }
        
        ImGui::EndTable();
    }
}

void CameraControlPanel::renderCameraSettings() {
    if (!ImGui::Begin("Camera Settings", &show_settings)) {
        ImGui::End();
        return;
    }

    for (size_t i = 0; i < camera_manager.getCameraCount(); i++) {
        const auto& camera = camera_manager.getCamera(i);
        
        if (ImGui::TreeNode(("Camera " + std::to_string(i)).c_str())) {
            // Exposure control
            int exposure = camera.params.exposure;
            if (ImGui::SliderInt("Exposure", &exposure, 0, 100000)) {
                try {
                    camera_manager.updateExposure(i, exposure);
                } catch (const evt::CameraException& e) {
                    // Handle error
                }
            }
            
            // Gain control
            int gain = camera.params.gain;
            if (ImGui::SliderInt("Gain", &gain, 0, 100)) {
                try {
                    camera_manager.updateGain(i, gain);
                } catch (const evt::CameraException& e) {
                    // Handle error
                }
            }
            
            // Frame rate control
            int frame_rate = camera.params.frame_rate;
            if (ImGui::SliderInt("Frame Rate", &frame_rate, 1, 120)) {
                try {
                    camera_manager.updateFrameRate(i, frame_rate);
                } catch (const evt::CameraException& e) {
                    // Handle error
                }
            }
            
            ImGui::TreePop();
        }
    }
    
    ImGui::End();
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