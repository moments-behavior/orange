#ifndef ORANGE_PROJECT
#define ORANGE_PROJECT
#include "camera.h"
#include "json.hpp"
#include "network_base.h"
#include "video_capture.h"
#include <sys/stat.h>
#include <unistd.h>
using json = nlohmann::json;

struct ConnectedServer {
    char name[80];
    uint8_t ip_add[4];
    uint16_t port;
    ENetPeer *peer;
    int num_cameras;
    FetchGame::ManagerState server_state;
    bool connected;
};

void prepare_application_folders(std::string orange_root_dir_str);
void intialize_servers(ConnectedServer *my_servers);
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
void client_send_bringup_message(EnetContext *enet_context,
                                 flatbuffers::FlatBufferBuilder *builder,
                                 ENetPeer *server_connection, int cam_count,
                                 FetchGame::ManagerState server_state);
void client_send_state_update_message(EnetContext *enet_context,
                                      flatbuffers::FlatBufferBuilder *builder,
                                      ENetPeer *server_connection,
                                      FetchGame::ManagerState server_state);
void host_broadcast_open_cameras(flatbuffers::FlatBufferBuilder *builder,
                                 EnetContext *server,
                                 std::string config_file_name);
void host_broadcast_start_threads(flatbuffers::FlatBufferBuilder *builder,
                                  EnetContext *server,
                                  std::string record_folder_name,
                                  std::string encoder_basic_setup);
void host_broadcast_set_start_ptp(flatbuffers::FlatBufferBuilder *builder,
                                  EnetContext *server,
                                  unsigned long long ptp_global_time);
void host_broadcast_test_focus(flatbuffers::FlatBufferBuilder *builder,
                               EnetContext *server);
void host_broadcast_setfocus(flatbuffers::FlatBufferBuilder *builder,
                             EnetContext *server, const char *serial,
                             int focus_value);
void host_broadcast_start_stream(flatbuffers::FlatBufferBuilder *builder,
                                 EnetContext *server,
                                 unsigned long long ptp_global_time);

#endif
