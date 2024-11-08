#include "main_window.h"
#include "camera_manager.h"
#include "fs_utils.hpp"
#include <memory>

int main(int argc, char** args) {
    // Initialize paths and filesystem
    if (!fs_utils::initialize_directories()) {
        return 1;
    }
    
    // Create main application objects
    std::unique_ptr<MainWindow> window = std::make_unique<MainWindow>();
    std::unique_ptr<evt::CameraManager> camera_manager = 
        std::make_unique<evt::CameraManager>();
    
    // Initialize GUI
    window->initialize();
    
    // Main loop
    while (!window->shouldClose()) {
        window->render();
    }
    
    // Cleanup
    window->cleanup();
    
    return 0;
}