#include "main_window.h"
#include "camera_manager.h"
#include "fs_utils.hpp"
#include "NvEncoder/Logger.h"
#include "gx_helper.h"
#include <memory>
#include <stdexcept>

// Define the global logger variable with a console logger
simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger(INFO, true);

int main(int argc, char** args) {
    try {
        // Initialize paths and filesystem
        if (!fs_utils::initialize_directories()) {
            LOG(ERROR) << "Failed to initialize directories";
            return 1;
        }
        
        LOG(INFO) << "Starting Orange application...";
        
        // Initialize GLFW first
        glfwSetErrorCallback(gx_glfw_error_callback);
        if (!glfwInit()) {
            LOG(ERROR) << "Failed to initialize GLFW";
            return 1;
        }
        LOG(INFO) << "GLFW initialized successfully";

        // Create main application objects
        std::unique_ptr<MainWindow> window;
        try {
            window = std::make_unique<MainWindow>();
            LOG(INFO) << "MainWindow created successfully";
        } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to create MainWindow: " << e.what();
            glfwTerminate();  // Clean up GLFW
            return 1;
        }
        
        // Initialize GUI
        try {
            window->initialize();
            LOG(INFO) << "GUI initialized successfully";
        } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to initialize GUI: " << e.what();
            glfwTerminate();  // Clean up GLFW
            return 1;
        }
        
        // Main loop
        while (!window->shouldClose()) {
            try {
                window->render();
            } catch (const std::exception& e) {
                LOG(ERROR) << "Error during render: " << e.what();
                break;
            }
        }
        
        // Cleanup
        try {
            window->cleanup();
            LOG(INFO) << "Application shutting down normally";
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error during cleanup: " << e.what();
        }
        
        // Clean up GLFW
        glfwTerminate();
        
        // Clean up logger
        delete logger;
        logger = nullptr;
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Unhandled exception in main: " << e.what();
        glfwTerminate();  // Ensure GLFW cleanup happens
        return 1;
    }
}