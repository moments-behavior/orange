#include <iostream>
#include <thread>
#include <filesystem>
#include <iostream>
#include "network_base.h"
#include "thread.h"
#include "types.h"
#include <cstring>
#include "video_capture.h"
#include "NvEncoder/NvCodecUtils.h"
#include "project.h"
#include "video_capture.h"
#include "fetch_generated.h"


simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();



void quit_process(bool error = false, const std::string &reason = "") {
    enet_deinitialize(); 
    //Show console reason before exit
    if (error) {
        std::cout << reason << std::endl;
        system("PAUSE");
        exit(-1);
    }
}


bool start_camera_thread(std::vector<std::thread>& camera_threads, CameraParams* cameras_params, CameraEmergent* ecams, CameraControl* camera_control, CameraEachSelect* cameras_select, GigEVisionDeviceInfo* device_info, int num_cameras, PTPParams* ptp_params)
{
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split(cwd, delimiter);

    std::vector<std::string> camera_config_files;
    std::vector<std::string> camera_config_names;
    int evt_buffer_size {100};

    printf("Start camera thread \n");
    cameras_params = new CameraParams[num_cameras];
    cameras_select = new CameraEachSelect[num_cameras];

    // load camera configs
    std::string start_folder_name = "/home/" + tokenized_path[2] + "/exp"; 
    std::string camera_config_dir = start_folder_name + "/5_camera_with_names/";

    for (const auto &entry : std::filesystem::directory_iterator(camera_config_dir))
    {
        camera_config_files.push_back(entry.path().string());
    }
    std::sort(camera_config_files.begin(), camera_config_files.end());
    for (auto &camera_serial : camera_config_files) {
        // get the serial number
        std::string delimiter = "/";
        std::vector<std::string> tokenized_path = string_split(camera_serial, delimiter);
        camera_config_names.push_back(tokenized_path.back());
    }


    for (int i = 0; i < num_cameras; i++)
    {
        cameras_params[i].camera_serial.append(device_info[i].serialNumber);
        auto it = std::find(camera_config_names.begin(), camera_config_names.end(), cameras_params[i].camera_serial + ".json");  
        if (it != camera_config_names.end()) {
            auto config_idx = std::distance(camera_config_names.begin(), it);
            std::cout << "Load camera json file: " << camera_config_files[config_idx] << std::endl;
            load_camera_json_config_files(camera_config_files[config_idx], &cameras_params[i], i, num_cameras); 
        }
    }

    ecams = new CameraEmergent[num_cameras];
    for (int i = 0; i < num_cameras; i++)
    {
        open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
    }

    for (int i = 0; i < num_cameras; i++)
    {               
        camera_open_stream(&ecams[i].camera);
        ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], evt_buffer_size);

        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
        {
            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
        }
    }
    
    camera_control->record_video = true; 
    camera_control->subscribe = true;
    camera_control->sync_camera = true;
    std::string encoder_setup = "-codec h264 -preset p1 -fps " + std::to_string(cameras_params[0].frame_rate);
    std::string folder_string = current_date_time();
    std::string folder_name = "/home/" + tokenized_path[2] + "/Videos/" + folder_string;    

    // Creating a directory to save recorded video;
    if (mkdir(folder_name.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return false;
    }
    else
    {
        std::cout << "Recorded video saves to : " << folder_name << std::endl;
    }

    for (int i = 0; i < num_cameras; i++)
    {
        ptp_camera_sync(&ecams[i].camera);
    }

    for (int i = 0; i < num_cameras; i++)
    {
        cameras_select->stream_on = false;
    }

    for (int i = 0; i < num_cameras; i++)
    {
        camera_threads.push_back(std::thread(&aquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, nullptr, encoder_setup, folder_name, ptp_params));
    }

    return true;
}


int main(int argc, char *argv[])
{
    if (enet_initialize() != 0)
    {
        quit_process(true, "ENET failed to initialize!");
    }

    ENetPeer* server_connection;
    EnetContext client;
    if(enet_initialize(&client, 0, 1)) {
        printf("Network Initialized!\n");
        server_connection = connect_peer(&client, 10, 123, 1, 142, 3333);
    }

    f32 last_time = tick();
    f32 current_time = tick();

    int max_cameras = 20;
    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);
    std::cout << "available no of cameras: " << cam_count << std::endl;
    
    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    CameraControl *camera_control = new CameraControl;
    PTPParams* ptp_params = new PTPParams{0, 0};
    CameraEachSelect *cameras_select;

    bool quit_recording = false;

    while(!quit_recording) {
        current_time = tick();
        //Handle All Incoming Packets and Send any enqued packets, does this need to be on another thread?
        service_network(&client, current_time - last_time, [&](const ENetEvent& evnt)
        {
            switch (evnt.type)
                {
                //New connection request or an existing peer accepted our connection request
                case ENET_EVENT_TYPE_CONNECT:
                    {
                        if (evnt.peer == server_connection)
                        {
                            printf("Network: Successfully connected to server! \n");

                            //Send a 'hello' packet
                            char* text_data = "Hellooo!";
                            ENetPacket* packet = enet_packet_create(text_data, strlen(text_data) + 1, 0);
                            enet_peer_send(server_connection, 0, packet);
                        }	
                    }
                    break;


                //Server has sent us a new packet
                case ENET_EVENT_TYPE_RECEIVE:
                    {
                        printf ("A packet of length %u containing %s was received from %s on channel %u.\n",
                                evnt.packet -> dataLength,
                                evnt.packet -> data,
                                evnt.peer -> data,
                                evnt.channelID);

                        uint8_t* buffer_pointer = evnt.packet->data;
                        auto server_control = FetchGame::GetServer(buffer_pointer);
                        auto server_signal = server_control->control();

                        if (server_signal == FetchGame::ServerControl_START)
                        {
                            printf("Start camera thread...");
                            // if(start_camera_thread(camera_threads, cameras_params, ecams, camera_control, cameras_select, device_info, cam_count, ptp_params)) {
                            //     printf("Camera threads started...\n");
                            // };
                        } 

                        enet_packet_destroy(evnt.packet);
                    }
                    break;

                //Server has disconnected
                case ENET_EVENT_TYPE_DISCONNECT:
                    {
                        printf("Network: Server has disconnected!");
                    }
                    break;
                }
        });
        last_time = current_time;
        sleep(1);
    }

    quit_process();
    return 0;
}