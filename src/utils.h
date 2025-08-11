#ifndef ORANGE_UTILS
#define ORANGE_UTILS
#include "camera.h"
#include "enet_fb_helpers.h"
#include "fetch_generated.h"
#include "json.hpp"
#include "video_capture.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <stdint.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

#define CHECK(call)                                                            \
    do {                                                                       \
        const cudaError_t error_code = call;                                   \
        if (error_code != cudaSuccess) {                                       \
            printf("CUDA Error:\n");                                           \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error code: %d\n", error_code);                        \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

inline float tick() {
    struct timespec ts;
    uint32_t res = clock_gettime(CLOCK_MONOTONIC, &ts);
    if (res == -1) {
        return 0;
    }
    return ((float)((ts.tv_sec * 1e9) + ts.tv_nsec)) / (float)1.0e9;
}

inline int find_cfg_index(const std::vector<std::string> &folders,
                          const char *base) {
    for (int i = 0; i < (int)folders.size(); ++i) {
        if (std::filesystem::path(folders[i]).filename() == base)
            return i;
    }
    return -1;
}

// Increment value with a lock and return the previous value
inline uint64_t sync_fetch_and_add(volatile uint64_t *x, uint64_t by) {
    // NOTE(dd): we're using a gcc/clang compiler extension to do this
    // because mutexes were for some reason slower
    return __sync_fetch_and_add(x, by);
}

inline NppStreamContext make_npp_stream_context(int device_id,
                                                cudaStream_t stream) {
    NppStreamContext ctx = {};
    cudaDeviceProp prop;

    // Set device
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&prop, device_id);

    ctx.hStream = stream;
    ctx.nCudaDeviceId = device_id;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;

    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                           cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                           cudaDevAttrComputeCapabilityMinor, device_id);

    unsigned int stream_flags = 0;
    cudaStreamGetFlags(stream, &stream_flags);
    ctx.nStreamFlags = stream_flags;

    // DO NOT touch ctx.nReserved0

    return ctx;
}

void prepare_application_folders(std::string orange_root_dir_str);
std::vector<std::string> string_split(std::string s, std::string delimiter);
std::vector<std::string> string_split_char(char *string_c,
                                           std::string delimiter);
void load_camera_json_config_files(std::string file_name,
                                   CameraParams *camera_params,
                                   CameraEachSelect *camera_select,
                                   int camera_id, int num_cameras);
std::string get_current_time_milliseconds();
std::string get_current_date();
std::string get_current_date_time();
std::string format_elapsed_time(std::chrono::seconds elapsed_seconds);
void init_galvo_camera_params(CameraParams *camera_params, int camera_id,
                              int num_cameras, int gain, int exposure);
void init_65MP_camera_params_mono(CameraParams *camera_params, int camera_id,
                                  int num_cameras, int gain, int exposure,
                                  int gpu_id, int frame_rate);
void init_65MP_camera_params_color(CameraParams *camera_params, int camera_id,
                                   int num_cameras, int gain, int exposure,
                                   int gpu_id, int frame_rate);
void init_7MP_camera_params_color(CameraParams *camera_params, int camera_id,
                                  int num_cameras, int gain, int exposure,
                                  int gpu_id, int frame_rate);
void init_7MP_camera_params_mono(CameraParams *camera_params, int camera_id,
                                 int num_cameras, int gain, int exposure,
                                 int gpu_id, int frame_rate);
bool make_folder(std::string folder_name);
void update_camera_configs(std::vector<std::string> &camera_config_files,
                           std::string input_folder);
void select_cameras_have_configs(std::vector<std::string> &camera_config_files,
                                 GigEVisionDeviceInfo *device_info,
                                 std::vector<bool> &check, int cam_count);
bool set_camera_params(CameraParams *camera_params,
                       CameraEachSelect *camera_select,
                       GigEVisionDeviceInfo *device_info,
                       std::vector<std::string> &camera_config_files,
                       int camera_idx, int num_cameras);
void allocate_camera_frame_buffers(CameraEmergent *ecams,
                                   CameraParams *cameras_params,
                                   int evt_buffer_size, int num_cameras);

inline void send_message(FBMessageSender &sender, uint32_t peer_id,
                         FetchGame::SignalType signal_type) {
    sender.to_peer(peer_id, [&](flatbuffers::FlatBufferBuilder &b) {
        b.Clear();
        FetchGame::ServerBuilder sb(b);
        sb.add_signal_type(signal_type);
        b.Finish(sb.Finish());
    });
}

// Convenience: send to a peer by its name (e.g., "indigo")
inline void send_message_to_indigo(FBMessageSender &sender,
                                   const PeerRegistry &reg,
                                   const std::string &peer_name,
                                   FetchGame::SignalType signal_type) {
    if (uint32_t pid = reg.get_pid_by_name(peer_name)) {
        send_message(sender, pid, signal_type);
    }
}

#endif // ORANGE_THREADS
