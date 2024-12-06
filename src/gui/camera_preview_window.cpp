#include "camera_preview_window.h"
#include "gx_helper.h"
#include <GL/gl.h>

CameraPreviewWindow::CameraPreviewWindow(evt::CameraManager& camera_mgr)
    : camera_manager_(camera_mgr) {
    preview_enabled_.resize(camera_mgr.getCameraCount(), false);
}

CameraPreviewWindow::~CameraPreviewWindow() {
    // Clean up OpenGL textures
    for (auto& tex : camera_textures_) {
        if (tex.texture_id != 0) {
            glDeleteTextures(1, &tex.texture_id);
        }
    }
}

void CameraPreviewWindow::createTexture(size_t camera_idx, int width, int height) {
    if (camera_idx >= camera_textures_.size()) {
        camera_textures_.resize(camera_idx + 1);
    }
    
    auto& tex = camera_textures_[camera_idx];
    
    if (tex.texture_id != 0) {
        glDeleteTextures(1, &tex.texture_id);
    }

    glGenTextures(1, &tex.texture_id);
    glBindTexture(GL_TEXTURE_2D, tex.texture_id);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, 
                 GL_UNSIGNED_BYTE, nullptr);

    tex.width = width;
    tex.height = height;
}

void CameraPreviewWindow::updateFrame(size_t camera_idx, const void* data, 
                                    int width, int height) {
    if (!preview_enabled_[camera_idx]) return;
    
    if (camera_idx >= camera_textures_.size() || 
        camera_textures_[camera_idx].texture_id == 0 ||
        camera_textures_[camera_idx].width != width ||
        camera_textures_[camera_idx].height != height) {
        createTexture(camera_idx, width, height);
    }

    auto& tex = camera_textures_[camera_idx];
    glBindTexture(GL_TEXTURE_2D, tex.texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB,
                    GL_UNSIGNED_BYTE, data);
}

void CameraPreviewWindow::render() {
    ImGui::BeginChild("Camera Feeds");
    
    // Calculate grid layout
    int cam_count = camera_manager_.getCameraCount();
    int cols = std::min(2, cam_count); // Max 2 columns
    
    for (size_t i = 0; i < cam_count; i++) {
        if (!preview_enabled_[i]) continue;

        const auto& tex = camera_textures_[i];
        if (tex.texture_id == 0) continue;

        if (i > 0 && (i % cols) != 0) ImGui::SameLine();
        
        // Display camera feed
        ImGui::Image((void*)(intptr_t)tex.texture_id, 
                    ImVec2(tex.width, tex.height));
    }
    
    ImGui::EndChild();
}

void CameraPreviewWindow::setPreviewEnabled(size_t camera_idx, bool enabled) {
    if (camera_idx >= preview_enabled_.size()) {
        preview_enabled_.resize(camera_idx + 1, false);
    }
    preview_enabled_[camera_idx] = enabled;
}