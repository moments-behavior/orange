// cuda_context_manager.h - Complete CUDA debugging and context management
#ifndef CUDA_CONTEXT_MANAGER_H
#define CUDA_CONTEXT_MANAGER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <iomanip>

// Thread-safe logging mutex
static std::mutex g_debug_log_mutex;

// Enhanced CUDA context logging
#define CUDA_CTX_LOG(msg) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_CTX] [Thread " << std::this_thread::get_id() << "] " \
              << msg << " | Current context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Enhanced CUDA runtime logging with device info
#define CUDA_RT_LOG(msg) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    int current_device = -1; \
    cudaGetDevice(&current_device); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_RT] [Thread " << std::this_thread::get_id() << "] " \
              << msg << " | Device: " << current_device \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Resource allocation/deallocation logging
#define CUDA_RESOURCE_LOG(action, ptr, size) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    int current_device = -1; \
    cudaGetDevice(&current_device); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_RES] [Thread " << std::this_thread::get_id() << "] " \
              << action << " | Ptr: " << static_cast<void*>(ptr) \
              << " | Size: " << size \
              << " | Device: " << current_device \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// TensorRT specific logging
#define TENSORRT_LOG(msg) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[TENSORRT] [Thread " << std::this_thread::get_id() << "] " \
              << msg << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Worker entry lifecycle logging
#define WORKER_ENTRY_LOG(action, entry) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    std::cout << "[WORKER_ENTRY] [Thread " << std::this_thread::get_id() << "] " \
              << action << " | Entry: " << static_cast<void*>(entry) \
              << " | d_image: " << static_cast<void*>((entry) ? (entry)->d_image : nullptr) \
              << " | ref_count: " << ((entry) ? (entry)->ref_count.load() : -1) << std::endl; \
} while(0)

// Stream logging
#define CUDA_STREAM_LOG(action, stream) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_STREAM] [Thread " << std::this_thread::get_id() << "] " \
              << action << " | Stream: " << static_cast<void*>(stream) \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Enhanced CUDA error checking with context info
#define CUDA_CHECK_WITH_CTX(call, location) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        int current_device = -1; \
        cudaGetDevice(&current_device); \
        CUcontext current_ctx = nullptr; \
        cuCtxGetCurrent(&current_ctx); \
        std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
        std::cerr << "[CUDA ERROR] " << location \
                  << " | Thread: " << std::this_thread::get_id() \
                  << " | Device: " << current_device \
                  << " | Context: " << static_cast<void*>(current_ctx) \
                  << " | Error: " << cudaGetErrorString(err) \
                  << " | Call: " << #call << std::endl; \
        abort(); \
    } \
} while(0)

// RAII CUDA Context Manager
class CudaContextManager {
public:
    explicit CudaContextManager(CUcontext context, const char* location = "unknown") 
        : context_(context), location_(location), pushed_(false) {
        if (context_) {
            CUresult result = cuCtxPushCurrent(context_);
            if (result == CUDA_SUCCESS) {
                pushed_ = true;
                #ifdef DEBUG_CUDA_CONTEXT
                std::lock_guard<std::mutex> lock(g_debug_log_mutex);
                std::cout << "[CTX_MANAGER] Pushed context " << static_cast<void*>(context_) 
                          << " at " << location_ << " (thread " << std::this_thread::get_id() << ")" << std::endl;
                #endif
            } else {
                std::lock_guard<std::mutex> lock(g_debug_log_mutex);
                std::cerr << "[CTX_MANAGER] ERROR: Failed to push context " << static_cast<void*>(context_) 
                          << " at " << location_ << " (thread " << std::this_thread::get_id() << ")" << std::endl;
            }
        }
    }
    
    ~CudaContextManager() {
        if (pushed_) {
            CUcontext popped_context;
            CUresult result = cuCtxPopCurrent(&popped_context);
            if (result == CUDA_SUCCESS) {
                #ifdef DEBUG_CUDA_CONTEXT
                std::lock_guard<std::mutex> lock(g_debug_log_mutex);
                std::cout << "[CTX_MANAGER] Popped context " << static_cast<void*>(popped_context) 
                          << " at " << location_ << " (thread " << std::this_thread::get_id() << ")" << std::endl;
                #endif
                if (popped_context != context_) {
                    std::lock_guard<std::mutex> lock(g_debug_log_mutex);
                    std::cerr << "[CTX_MANAGER] WARNING: Context mismatch! Expected " 
                              << static_cast<void*>(context_) << " got " << static_cast<void*>(popped_context) 
                              << " at " << location_ << std::endl;
                }
            } else {
                std::lock_guard<std::mutex> lock(g_debug_log_mutex);
                std::cerr << "[CTX_MANAGER] ERROR: Failed to pop context at " << location_ 
                          << " (thread " << std::this_thread::get_id() << ")" << std::endl;
            }
        }
    }
    
    // Non-copyable, non-movable
    CudaContextManager(const CudaContextManager&) = delete;
    CudaContextManager& operator=(const CudaContextManager&) = delete;
    CudaContextManager(CudaContextManager&&) = delete;
    CudaContextManager& operator=(CudaContextManager&&) = delete;
    
    bool is_valid() const { return pushed_; }
    
private:
    CUcontext context_;
    const char* location_;
    bool pushed_;
};

// Convenience macro for RAII context management
#define CUDA_CONTEXT_SCOPE(ctx) CudaContextManager ctx_manager(ctx, __FUNCTION__)

// Alternative macro with custom location
#define CUDA_CONTEXT_SCOPE_AT(ctx, location) CudaContextManager ctx_manager(ctx, location)

// Function to validate CUDA context consistency
inline bool validate_cuda_context(const char* location, CUcontext expected_ctx = nullptr) {
    CUcontext current_ctx = nullptr;
    CUresult result = cuCtxGetCurrent(&current_ctx);
    
    if (result != CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_debug_log_mutex);
        std::cerr << "[CTX_VALIDATION] " << location 
                  << " | Thread: " << std::this_thread::get_id()
                  << " | ERROR: Failed to get current context" << std::endl;
        return false;
    }
    
    if (expected_ctx && current_ctx != expected_ctx) {
        std::lock_guard<std::mutex> lock(g_debug_log_mutex);
        std::cerr << "[CTX_VALIDATION] " << location 
                  << " | Thread: " << std::this_thread::get_id()
                  << " | ERROR: Context mismatch. Expected: " << static_cast<void*>(expected_ctx)
                  << " | Current: " << static_cast<void*>(current_ctx) << std::endl;
        return false;
    }
    
    return true;
}

// Function to log detailed CUDA memory info
inline void log_cuda_memory_info(const char* location) {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    std::lock_guard<std::mutex> lock(g_debug_log_mutex);
    if (err == cudaSuccess) {
        std::cout << "[CUDA_MEM] " << location 
                  << " | Thread: " << std::this_thread::get_id()
                  << " | Free: " << (free_mem / 1024 / 1024) << "MB"
                  << " | Total: " << (total_mem / 1024 / 1024) << "MB"
                  << " | Used: " << ((total_mem - free_mem) / 1024 / 1024) << "MB" << std::endl;
    } else {
        std::cerr << "[CUDA_MEM] " << location 
                  << " | Thread: " << std::this_thread::get_id()
                  << " | ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}

#endif // CUDA_CONTEXT_MANAGER_H