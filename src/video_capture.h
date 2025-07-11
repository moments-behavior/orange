// src/video_capture.h
#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE
#include "thread.h"
#include "camera.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "network_base.h"
#include "common.hpp" // For pose::Object
#include <cuda_runtime.h>
#include "NvEncoder/NvCodecUtils.h"

typedef struct {
    unsigned char* d_image;
    int width;
    int height;
    int pixelFormat;
    unsigned long long timestamp;
    unsigned long long frame_id;
    uint64_t timestamp_sys;
    
    // YOLO detection fields
    std::vector<pose::Object> detections;
    bool has_detections;
    std::atomic<bool> detections_ready;
    
    // Reference counting for memory management
    std::atomic<int> ref_count;
    
    // GPU Direct optimization fields
    bool gpu_direct_mode = false;
    bool owns_memory = true;
    
    // Camera buffer management (only used when gpu_direct_mode = true)
    void* camera_buffer_ptr = nullptr;
    Emergent::CEmergentCamera* camera_instance = nullptr;
    Emergent::CEmergentFrame* camera_frame_struct = nullptr;
    
    // Event for synchronization between workers. Now a pointer.
    cudaEvent_t* event_ptr; 

    // New event specifically for YOLO completion
    cudaEvent_t* yolo_completion_event; 

} WORKER_ENTRY;

enum PictureSaveState {
    State_Frame_Idle = 0,
    State_Write_New_Frame = 1
};

struct CameraControl
{
    bool open = false;
    bool subscribe = false;
    bool stop_record = false;
    bool record_video = false;
    bool sync_camera = false;
};

struct CameraResources {
    static const int ACQUIRE_WORK_ENTRIES_MAX = 120;
    static const int EVENT_POOL_SIZE = 256;

    WORKER_ENTRY* worker_entry_pool = nullptr;
    SafeQueue<WORKER_ENTRY*>* free_entries_queue = nullptr;
    SafeQueue<WORKER_ENTRY*>* recycle_queue = nullptr;

    std::vector<cudaEvent_t> event_pool;
    std::vector<cudaEvent_t> yolo_event_pool;
    SafeQueue<cudaEvent_t*>* free_events_queue = nullptr;
    SafeQueue<cudaEvent_t*>* yolo_events_queue = nullptr;

    CameraResources() = default;
    CameraResources(const CameraResources&) = delete;
    CameraResources& operator=(const CameraResources&) = delete;

    CameraResources(CameraResources&& other) noexcept {
        worker_entry_pool = other.worker_entry_pool;
        free_entries_queue = other.free_entries_queue;
        recycle_queue = other.recycle_queue;
        event_pool = std::move(other.event_pool);
        yolo_event_pool = std::move(other.yolo_event_pool);
        free_events_queue = other.free_events_queue;
        yolo_events_queue = other.yolo_events_queue;
        other.worker_entry_pool = nullptr;
        other.free_entries_queue = nullptr;
        other.recycle_queue = nullptr;
        other.free_events_queue = nullptr;
        other.yolo_events_queue = nullptr;
    }

    CameraResources& operator=(CameraResources&& other) noexcept {
        if (this != &other) {
            cleanup();
            worker_entry_pool = other.worker_entry_pool;
            free_entries_queue = other.free_entries_queue;
            recycle_queue = other.recycle_queue;
            event_pool = std::move(other.event_pool);
            yolo_event_pool = std::move(other.yolo_event_pool);
            free_events_queue = other.free_events_queue;
            yolo_events_queue = other.yolo_events_queue;
            other.worker_entry_pool = nullptr;
            other.free_entries_queue = nullptr;
            other.recycle_queue = nullptr;
            other.free_events_queue = nullptr;
            other.yolo_events_queue = nullptr;
        }
        return *this;
    }

    void initialize(int gpu_id, size_t frame_size) {
        ck(cudaSetDevice(gpu_id));
        
        worker_entry_pool = new WORKER_ENTRY[ACQUIRE_WORK_ENTRIES_MAX];
        for (int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
            ck(cudaMalloc(&worker_entry_pool[i].d_image, frame_size));
        }
        
        free_entries_queue = new SafeQueue<WORKER_ENTRY*>();
        for (int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
            free_entries_queue->push(&worker_entry_pool[i]);
        }
        
        recycle_queue = new SafeQueue<WORKER_ENTRY*>();
        
        event_pool.resize(EVENT_POOL_SIZE);
        free_events_queue = new SafeQueue<cudaEvent_t*>();
        for (int i = 0; i < EVENT_POOL_SIZE; ++i) {
            ck(cudaEventCreateWithFlags(&event_pool[i], cudaEventDisableTiming));
            free_events_queue->push(&event_pool[i]);
        }

        yolo_event_pool.resize(EVENT_POOL_SIZE);
        yolo_events_queue = new SafeQueue<cudaEvent_t*>();
        for (int i = 0; i < EVENT_POOL_SIZE; ++i) {
            ck(cudaEventCreateWithFlags(&yolo_event_pool[i], cudaEventDisableTiming));
            yolo_events_queue->push(&yolo_event_pool[i]);
        }
    }

    void cleanup() {
        if (worker_entry_pool) {
            for (int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
                if (worker_entry_pool[i].d_image) cudaFree(worker_entry_pool[i].d_image);
            }
            delete[] worker_entry_pool;
            worker_entry_pool = nullptr;
        }

        if (free_entries_queue) { delete free_entries_queue; free_entries_queue = nullptr; }
        if (recycle_queue) { delete recycle_queue; recycle_queue = nullptr; }
        
        if (free_events_queue) { delete free_events_queue; free_events_queue = nullptr; }
        for (auto& event : event_pool) {
            if (event) cudaEventDestroy(event);
        }
        event_pool.clear();

        if (yolo_events_queue) { delete yolo_events_queue; yolo_events_queue = nullptr; }
        for (auto& event : yolo_event_pool) {
            if (event) cudaEventDestroy(event);
        }
        yolo_event_pool.clear();
    }
};


struct CameraEachSelect
{
    bool stream_on = true;
    bool record = false;
    bool yolo = false;
    int downsample = 1;
    PictureSaveState frame_save_state = State_Frame_Idle;
    std::string frame_save_format;
    std::string frame_save_name;
    int pictures_counter = 0;
    bool selected_to_save = false;
    std::string picture_save_folder;
    const char* yolo_model;
    // New flags for configurable YOLO output:
    bool send_yolo_via_ipc = false; // Default to false
    bool send_yolo_via_enet = false;  // Default to true (current behavior)
};

struct CameraState
{
    int camera_return = 0;
    unsigned short id_prev = 0;
    unsigned short dropped_frames = 0;
    unsigned int frames_recd = 0;
    unsigned long long frame_count = 0;
};

struct PTPState
{
    int ptp_offset;
    int ptp_offset_sum=0;
    int ptp_offset_prev=0;
    unsigned int ptp_time_low;
    unsigned int ptp_time_high;
    unsigned int ptp_time_plus_delta_to_start_low;
    unsigned int ptp_time_plus_delta_to_start_high;
    unsigned long long ptp_time_delta_sum = 0;
    unsigned long long ptp_time_delta;
    unsigned long long ptp_time;
    unsigned long long ptp_time_prev;
    unsigned long long ptp_time_countdown;
    unsigned long long frame_ts;
    unsigned long long frame_ts_prev;
    unsigned long long frame_ts_delta;
    unsigned long long frame_ts_delta_sum = 0;
    unsigned long long ptp_time_plus_delta_to_start;
    char ptp_status[100];
    unsigned long ptp_status_sz_ret;
    unsigned int ptp_time_plus_delta_to_start_uint;
};

void report_statistics(CameraParams *camera_params, CameraState *camera_state, double time_diff);
void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam);
void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params, CameraParams *camera_params, CameraEmergent *ecam, unsigned int delay_in_second);
void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam);
bool try_start_timer();
bool try_stop_timer();
#endif