#include "main_window.h"
#include "camera_manager.h"
#include "fs_utils.hpp"
#include "NvEncoder/Logger.h"
#include <memory>

// Define the global logger variable with a console logger
// The parameters are: LogLevel (INFO by default), and true to print timestamps
simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger(INFO, true);

int main(int argc, char** args) {
    // Initialize paths and filesystem
    if (!fs_utils::initialize_directories()) {
        LOG(ERROR) << "Failed to initialize directories";
        return 1;
    }
    
    LOG(INFO) << "Starting Orange application...";
    
    // Create main application objects
    std::unique_ptr<MainWindow> window = std::make_unique<MainWindow>();
    std::unique_ptr<evt::CameraManager> camera_manager = 
        std::make_unique<evt::CameraManager>();
    
    // Initialize GUI
    window->initialize();
    LOG(INFO) << "GUI initialized successfully";
    
    // Main loop
    while (!window->shouldClose()) {
        window->render();
    }
    
    // Cleanup
    window->cleanup();
    LOG(INFO) << "Application shutting down normally";
    
    // Clean up logger
    delete logger;
    logger = nullptr;
    
    return 0;
}