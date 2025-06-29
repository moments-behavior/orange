// src/nvtx_profiling.h
#ifndef NVTX_PROFILING_H
#define NVTX_PROFILING_H

// NVTX profiling wrapper for conditional compilation
#ifdef ENABLE_NVTX_PROFILING
#include <nvtx3/nvToolsExt.h>

// Convenient macros for NVTX markers
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()

// RAII NVTX range for automatic pop
class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~NvtxRange() {
        nvtxRangePop();
    }
    
    // Non-copyable
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

// Convenient macro for RAII ranges
#define NVTX_RANGE(name) NvtxRange nvtx_range_##__LINE__(name)

// Colored ranges for better visualization
#define NVTX_RANGE_COLORED(name, color) \
    do { \
        nvtxEventAttributes_t attrib = {0}; \
        attrib.version = NVTX_VERSION; \
        attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        attrib.colorType = NVTX_COLOR_ARGB; \
        attrib.color = color; \
        attrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        attrib.message.ascii = name; \
        nvtxRangePushEx(&attrib); \
    } while(0)

// Predefined colors for different categories
#define NVTX_COLOR_CAMERA     0xFF00FF00  // Green - Camera operations
#define NVTX_COLOR_GPU_COPY   0xFF0000FF  // Blue - Memory operations
#define NVTX_COLOR_ENCODE     0xFFFF0000  // Red - Encoding operations
#define NVTX_COLOR_YOLO       0xFFFFFF00  // Yellow - YOLO inference
#define NVTX_COLOR_DISPLAY    0xFFFF00FF  // Magenta - Display operations
#define NVTX_COLOR_SYNC       0xFF800080  // Purple - Synchronization
#define NVTX_COLOR_QUEUE      0xFF808080  // Gray - Queue operations

// Specific category macros
#define NVTX_CAMERA(name)     NVTX_RANGE_COLORED(name, NVTX_COLOR_CAMERA)
#define NVTX_GPU_COPY(name)   NVTX_RANGE_COLORED(name, NVTX_COLOR_GPU_COPY)
#define NVTX_ENCODE(name)     NVTX_RANGE_COLORED(name, NVTX_COLOR_ENCODE)
#define NVTX_YOLO(name)       NVTX_RANGE_COLORED(name, NVTX_COLOR_YOLO)
#define NVTX_DISPLAY(name)    NVTX_RANGE_COLORED(name, NVTX_COLOR_DISPLAY)
#define NVTX_SYNC(name)       NVTX_RANGE_COLORED(name, NVTX_COLOR_SYNC)
#define NVTX_QUEUE(name)      NVTX_RANGE_COLORED(name, NVTX_COLOR_QUEUE)

#else
// No-op macros when NVTX is disabled
#define NVTX_RANGE_PUSH(name) do {} while(0)
#define NVTX_RANGE_POP() do {} while(0)
#define NVTX_RANGE(name) do {} while(0)
#define NVTX_RANGE_COLORED(name, color) do {} while(0)
#define NVTX_CAMERA(name) do {} while(0)
#define NVTX_GPU_COPY(name) do {} while(0)
#define NVTX_ENCODE(name) do {} while(0)
#define NVTX_YOLO(name) do {} while(0)
#define NVTX_DISPLAY(name) do {} while(0)
#define NVTX_SYNC(name) do {} while(0)
#define NVTX_QUEUE(name) do {} while(0)

class NvtxRange {
public:
    explicit NvtxRange(const char*) {}
};

#endif // ENABLE_NVTX_PROFILING

#endif // NVTX_PROFILING_H