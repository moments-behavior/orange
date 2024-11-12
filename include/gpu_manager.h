// gpu_manager.h
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace evt {

class GPUException : public std::runtime_error {
public:
    explicit GPUException(const std::string& msg, cudaError_t cuda_error = cudaSuccess)
        : std::runtime_error(msg), error_code(cuda_error) {}
    cudaError_t getErrorCode() const { return error_code; }
private:
    cudaError_t error_code;
};

struct GPUDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    bool supports_gpu_direct;
};

class GPUManager {
public:
    static GPUManager& getInstance() {
        static GPUManager instance;
        return instance;
    }

    // Initialize CUDA and get device information
    void initialize();
    
    // Check if GPU Direct is supported for a specific device
    bool isGPUDirectSupported(int device_id) const;
    
    // Get available GPU devices
    const std::vector<GPUDeviceInfo>& getAvailableDevices() const { return available_devices_; }
    
    // Select and configure a specific GPU
    void selectDevice(int device_id);
    
    // Get current GPU memory status
    void getMemoryInfo(size_t& free, size_t& total) const;
    
    // Verify GPU Direct compatibility with camera
    bool verifyGPUDirectCompatibility(const std::string& camera_serial) const;

    // Clean up resources
    void cleanup();

private:
    GPUManager() = default;
    ~GPUManager() = default;
    
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
    
    void checkCUDAError(cudaError_t error, const std::string& operation) const;
    bool initialized_ = false;
    std::vector<GPUDeviceInfo> available_devices_;
    int current_device_ = -1;
};

} // namespace evt
