// src/cuda_context_debug.h - Fixed version to match build script
#ifndef CUDA_CONTEXT_DEBUG_H
#define CUDA_CONTEXT_DEBUG_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <sstream>

// Thread-safe logging mutex
static std::mutex g_debug_log_mutex;

// Enable debug logging - match what the build script uses
#if defined(DEBUG_CUDA_CONTEXT) || defined(ENABLE_CUDA_DEBUG_LOGGING)

// Core context logging macro
#define CUDA_CTX_LOG(msg) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_CTX] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " \
              << msg << " | Current context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Enhanced runtime logging with device info
#define CUDA_RT_LOG(msg) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    int current_device = -1; \
    cudaGetDevice(&current_device); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_RT] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " \
              << msg << " | Device: " << current_device \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Encoder-specific logging
#define ENCODER_CTX_LOG(msg, frame_id) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    int current_device = -1; \
    cudaGetDevice(&current_device); \
    std::cout << "[ENCODER_CTX] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " \
              << "Frame " << frame_id << " | " << msg \
              << " | Device: " << current_device \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Memory operation logging
#define CUDA_MEM_LOG(action, ptr, size, frame_id) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_MEM] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " \
              << "Frame " << frame_id << " | " << action \
              << " | Ptr: " << static_cast<void*>(ptr) \
              << " | Size: " << size \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Stream synchronization logging
#define CUDA_SYNC_LOG(action, stream, frame_id) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_SYNC] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " \
              << "Frame " << frame_id << " | " << action \
              << " | Stream: " << static_cast<void*>(stream) \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

// Stream logging (for cases without frame_id)
#define CUDA_STREAM_LOG(action, stream) \
do { \
    std::lock_guard<std::mutex> lock(g_debug_log_mutex); \
    CUcontext current_ctx = nullptr; \
    cuCtxGetCurrent(&current_ctx); \
    std::cout << "[CUDA_STREAM] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " \
              << action << " | Stream: " << static_cast<void*>(stream) \
              << " | Context: " << static_cast<void*>(current_ctx) << std::endl; \
} while(0)

#else
// No-op macros when debugging is disabled
#define CUDA_CTX_LOG(msg) do {} while(0)
#define CUDA_RT_LOG(msg) do {} while(0)
#define ENCODER_CTX_LOG(msg, frame_id) do {} while(0)
#define CUDA_MEM_LOG(action, ptr, size, frame_id) do {} while(0)
#define CUDA_SYNC_LOG(action, stream, frame_id) do {} while(0)
#define CUDA_STREAM_LOG(action, stream) do {} while(0)
#endif

// Enhanced context push/pop wrappers with validation (always available)
inline CUresult cuCtxPushCurrentDebug(CUcontext ctx, const char* location, unsigned long long frame_id = 0) {
    CUcontext before_ctx = nullptr;
    cuCtxGetCurrent(&before_ctx);
    
    CUresult result = cuCtxPushCurrent(ctx);
    
    CUcontext after_ctx = nullptr;
    cuCtxGetCurrent(&after_ctx);
    
    #if defined(DEBUG_CUDA_CONTEXT) || defined(ENABLE_CUDA_DEBUG_LOGGING)
    std::lock_guard<std::mutex> lock(g_debug_log_mutex);
    std::cout << "[CTX_PUSH] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " 
              << location;
    if (frame_id > 0) std::cout << " (Frame " << frame_id << ")";
    std::cout << " | Before: " << static_cast<void*>(before_ctx)
              << " | Pushing: " << static_cast<void*>(ctx)
              << " | After: " << static_cast<void*>(after_ctx)
              << " | Result: " << (result == CUDA_SUCCESS ? "SUCCESS" : "FAILED") << std::endl;
    #endif
    
    return result;
}

inline CUresult cuCtxPopCurrentDebug(CUcontext* pctx, const char* location, unsigned long long frame_id = 0) {
    CUcontext before_ctx = nullptr;
    cuCtxGetCurrent(&before_ctx);
    
    CUresult result = cuCtxPopCurrent(pctx);
    
    CUcontext after_ctx = nullptr;
    cuCtxGetCurrent(&after_ctx);
    
    #if defined(DEBUG_CUDA_CONTEXT) || defined(ENABLE_CUDA_DEBUG_LOGGING)
    std::lock_guard<std::mutex> lock(g_debug_log_mutex);
    std::cout << "[CTX_POP] [Thread " << std::hex << std::this_thread::get_id() << std::dec << "] " 
              << location;
    if (frame_id > 0) std::cout << " (Frame " << frame_id << ")";
    std::cout << " | Before: " << static_cast<void*>(before_ctx)
              << " | Popped: " << static_cast<void*>(pctx ? *pctx : nullptr)
              << " | After: " << static_cast<void*>(after_ctx)
              << " | Result: " << (result == CUDA_SUCCESS ? "SUCCESS" : "FAILED") << std::endl;
    #endif
    
    return result;
}

// Enhanced error checking with context validation
inline void validateCudaOperation(const char* operation, const char* file, int line, unsigned long long frame_id = 0) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        CUcontext current_ctx = nullptr;
        cuCtxGetCurrent(&current_ctx);
        int device = -1;
        cudaGetDevice(&device);
        
        std::lock_guard<std::mutex> lock(g_debug_log_mutex);
        std::cerr << "[CUDA_ERROR] " << operation;
        if (frame_id > 0) std::cerr << " (Frame " << frame_id << ")";
        std::cerr << " | " << file << ":" << line
                  << " | Thread: " << std::hex << std::this_thread::get_id() << std::dec
                  << " | Device: " << device
                  << " | Context: " << static_cast<void*>(current_ctx)
                  << " | Error: " << cudaGetErrorString(err) << std::endl;
    }
}

// RAII CUDA Context Manager for compatibility with old code
class CudaContextManager {
public:
    explicit CudaContextManager(CUcontext context, const char* location = "unknown") 
        : context_(context), location_(location), valid_(false) {
        if (context_) {
            CUresult result = cuCtxPushCurrent(context_);
            valid_ = (result == CUDA_SUCCESS);
            #if defined(DEBUG_CUDA_CONTEXT) || defined(ENABLE_CUDA_DEBUG_LOGGING)
            if (valid_) {
                std::lock_guard<std::mutex> lock(g_debug_log_mutex);
                std::cout << "[CTX_SCOPE] Pushed context " << static_cast<void*>(context_) 
                          << " at " << location_ << " (thread " << std::hex << std::this_thread::get_id() << std::dec << ")" << std::endl;
            }
            #endif
        }
    }
    
    ~CudaContextManager() {
        if (valid_ && context_) {
            CUcontext popped_context;
            cuCtxPopCurrent(&popped_context);
            #if defined(DEBUG_CUDA_CONTEXT) || defined(ENABLE_CUDA_DEBUG_LOGGING)
            std::lock_guard<std::mutex> lock(g_debug_log_mutex);
            std::cout << "[CTX_SCOPE] Popped context " << static_cast<void*>(popped_context) 
                      << " at " << location_ << " (thread " << std::hex << std::this_thread::get_id() << std::dec << ")" << std::endl;
            #endif
        }
    }
    
    bool is_valid() const { return valid_; }
    
private:
    CUcontext context_;
    const char* location_;
    bool valid_;
    
    // Non-copyable
    CudaContextManager(const CudaContextManager&) = delete;
    CudaContextManager& operator=(const CudaContextManager&) = delete;
};

// Compatibility macros for old code
#define CUDA_CONTEXT_SCOPE_AT(context, location) \
    CudaContextManager ctx_manager(context, location)

#define CUDA_CONTEXT_SCOPE(context) \
    CudaContextManager ctx_manager(context, __FUNCTION__)

// Convenience macro for error checking
#define VALIDATE_CUDA_OP(op, frame_id) validateCudaOperation(op, __FILE__, __LINE__, frame_id)

// Function to dump complete CUDA state
inline void dumpCudaState(const char* location, unsigned long long frame_id = 0) {
    #if defined(DEBUG_CUDA_CONTEXT) || defined(ENABLE_CUDA_DEBUG_LOGGING)
    std::lock_guard<std::mutex> lock(g_debug_log_mutex);
    
    CUcontext current_ctx = nullptr;
    cuCtxGetCurrent(&current_ctx);
    
    int device = -1;
    cudaGetDevice(&device);
    
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    std::cout << "[CUDA_STATE] " << location;
    if (frame_id > 0) std::cout << " (Frame " << frame_id << ")";
    std::cout << " | Thread: " << std::hex << std::this_thread::get_id() << std::dec
              << " | Device: " << device
              << " | Context: " << static_cast<void*>(current_ctx)
              << " | Free mem: " << (free_bytes / 1024 / 1024) << "MB"
              << " | Total mem: " << (total_bytes / 1024 / 1024) << "MB" << std::endl;
    #endif
}

#endif // CUDA_CONTEXT_DEBUG_H