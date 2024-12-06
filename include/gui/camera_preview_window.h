#pragma once

#include <vector>
#include "camera_manager.h"
#include "imgui.h"
#include <memory>
#include <string>

class CameraPreviewWindow {
public:
    explicit CameraPreviewWindow(evt::CameraManager& camera_mgr);
    ~CameraPreviewWindow();

    void render();
    void updateFrame(size_t camera_idx, const void* data, int width, int height);
    bool isPreviewEnabled(size_t camera_idx) const { 
        return camera_idx < preview_enabled_.size() && preview_enabled_[camera_idx]; 
    }
    void setPreviewEnabled(size_t camera_idx, bool enabled);

private:
    struct CameraTexture {
        unsigned int texture_id{0};
        int width{0};
        int height{0};
    };

    void createTexture(size_t camera_idx, int width, int height);

    evt::CameraManager& camera_manager_;
    std::vector<CameraTexture> camera_textures_;
    std::vector<bool> preview_enabled_;
};