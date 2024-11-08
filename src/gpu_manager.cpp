#include "gpu_manager.h"
#include <sstream>
#include <iostream>

namespace evt {

void GPUManager::initialize() {
    if (initialized_) {
        return;
    }

    int device_count;
    checkCUDAError(cudaGetDeviceCount(&device_count),
        "Failed to get CUDA device count");

    if (device_count == 0) {
        throw GPUException("No CUDA devices found");
    }

    available_devices_.clear();
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        checkCUDAError(cudaGetDeviceProperties(&prop, i),
            "Failed to get device properties for GPU " + std::to_string(i));

        GPUDeviceInfo device;
        device.device_id = i;
        device.name = prop.name;
        device.compute_capability_major = prop.major;
        device.compute_capability_minor = prop.minor;
        
        // Check GPU Direct support
        device.supports_gpu_direct = prop.directManagedMemAccessFromHost;

        // Get memory information
        selectDevice(i);
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        device.free_memory = free;
        device.total_memory = total;

        available_devices_.push_back(device);
    }

    initialized_ = true;
}

void GPUManager::selectDevice(int device_id) {
    if (device_id >= static_cast<int>(available_devices_.size())) {
        throw GPUException("Invalid GPU device ID: " + std::to_string(device_id));
    }

    checkCUDAError(cudaSetDevice(device_id),
        "Failed to set CUDA device " + std::to_string(device_id));
    
    current_device_ = device_id;
    
    // Ensure device is in proper state
    checkCUDAError(cudaDeviceSynchronize(),
        "Failed to synchronize CUDA device");
}

bool GPUManager::isGPUDirectSupported(int device_id) const {
    if (!initialized_) {
        throw GPUException("GPU Manager not initialized");
    }

    if (device_id >= static_cast<int>(available_devices_.size())) {
        throw GPUException("Invalid GPU device ID");
    }

    return available_devices_[device_id].supports_gpu_direct;
}

void GPUManager::getMemoryInfo(size_t& free, size_t& total) const {
    if (current_device_ == -1) {
        throw GPUException("No GPU device selected");
    }

    checkCUDAError(cudaMemGetInfo(&free, &total),
        "Failed to get GPU memory info");
}

bool GPUManager::verifyGPUDirectCompatibility(const std::string& camera_serial) const {
    if (current_device_ == -1) {
        throw GPUException("No GPU device selected");
    }

    const auto& device = available_devices_[current_device_];
    
    // Log verification attempt
    std::cout << "Verifying GPU Direct compatibility for:" << std::endl
              << "Camera: " << camera_serial << std::endl
              << "GPU: " << device.name 
              << " (Device " << device.device_id << ")" << std::endl
              << "Compute Capability: " 
              << device.compute_capability_major << "."
              << device.compute_capability_minor << std::endl;

    // Check basic requirements
    if (!device.supports_gpu_direct) {
        std::cout << "GPU Direct not supported by this GPU" << std::endl;
        return false;
    }

    // Check memory availability
    size_t free_memory, total_memory;
    getMemoryInfo(free_memory, total_memory);
    
    // Require at least 1GB free memory for GPU Direct
    const size_t MIN_REQUIRED_MEMORY = 1024 * 1024 * 1024; // 1GB
    if (free_memory < MIN_REQUIRED_MEMORY) {
        std::cout << "Insufficient GPU memory for GPU Direct" << std::endl
                  << "Available: " << (free_memory / 1024 / 1024) << "MB"
                  << "Required: " << (MIN_REQUIRED_MEMORY / 1024 / 1024) << "MB"
                  << std::endl;
        return false;
    }

    return true;
}

void GPUManager::cleanup() {
    if (current_device_ != -1) {
        cudaDeviceReset();
    }
    initialized_ = false;
    current_device_ = -1;
    available_devices_.clear();
}

void GPUManager::checkCUDAError(cudaError_t error, const std::string& operation) const {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << operation << " failed: " 
            << cudaGetErrorString(error)
            << " (Error code: " << error << ")";
        throw GPUException(oss.str(), error);
    }
}

} // namespace evt