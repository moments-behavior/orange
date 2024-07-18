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
#include "acquire_frames_headless.h"
#include <signal.h>

#define evt_buffer_size 100
#define max_cameras 20

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void quit_process(bool error = false, const std::string &reason = "")
{
    enet_deinitialize();
    // Show console reason before exit
    if (error)
    {
        std::cout << reason << std::endl;
        system("PAUSE");
        exit(-1);
    }
}

bool open_cameras(CameraParams *cameras_params, CameraEmergent *ecams, CameraEachSelect *cameras_select, GigEVisionDeviceInfo *device_info, int num_cameras, std::string config_folder)
{
    std::vector<std::string> camera_config_files;
    update_camera_configs(camera_config_files, config_folder);

    for (int i = 0; i < num_cameras; i++)
    {
        set_camera_params(&cameras_params[i], &device_info[i], camera_config_files, i, num_cameras);
        open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
    }
    return true;
}


bool start_camera_thread(std::vector<std::thread> &camera_threads, CameraParams *cameras_params, CameraEmergent *ecams, CameraControl *camera_control, CameraEachSelect *cameras_select, 
    GigEVisionDeviceInfo *device_info, int num_cameras, PTPParams *ptp_params, std::string record_folder, std::string encoder_basic_setup)
{
    std::cout << "start camera sthread..." << std::endl;
    allocate_camera_frame_buffers(ecams, cameras_params, evt_buffer_size, num_cameras);
    camera_control->record_video = true;
    camera_control->subscribe = true;
    camera_control->sync_camera = true;

    // Creating a directory to save recorded video;
    if (mkdir(record_folder.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return false;
    }
    else
    {
        std::cout << "Recorded video saves to : " << record_folder << std::endl;
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
        std::string encoder_setup = encoder_basic_setup + std::to_string(cameras_params[i].frame_rate);
        camera_threads.push_back(std::thread(&acquire_frames_headless, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, encoder_setup, record_folder, ptp_params));
    }

    // wait for all camera ready
    while(ptp_params->ptp_counter!=num_cameras) {
        usleep(10);
    }

    return true;
}

bool quit_server = false;


static void interruptHandler(const int signal)
{
    printf("\nQuit Orange.\n");
    quit_server = true;
}

struct ManagerContext
{
    FetchGame::ManagerState state;
    bool quit;
};

struct RecordingContext {
    std::string record_folder;
    std::string encoder_basic_setup;
};

void create_camera_manager(int* cam_count, ManagerContext* manager_context, GigEVisionDeviceInfo* unsorted_device_info, GigEVisionDeviceInfo* device_info, std::string* config_folder, RecordingContext* recording_setup, PTPParams *ptp_params) 
{
    // TODO: selecting cameras 
    CameraEmergent *ecams;
    CameraParams *cameras_params;
    std::vector<std::thread> camera_threads;
    CameraEachSelect *cameras_select;
    CameraControl *camera_control = new CameraControl;

    manager_context->state = FetchGame::ManagerState_IDLE;
    while(!manager_context->quit) {
        switch (manager_context->state) {
            case FetchGame::ManagerState_CONNECT:
                *cam_count = scan_cameras(max_cameras, unsorted_device_info);
                std::cout << *cam_count << std::endl;
                sort_cameras_ip(unsorted_device_info, device_info, *cam_count);
                manager_context->state = FetchGame::ManagerState_CONNECTED;
                break;
            case FetchGame::ManagerState_OPENCAMERA:
                ecams = new CameraEmergent[*cam_count];
                cameras_params = new CameraParams[*cam_count];
                cameras_select = new CameraEachSelect[*cam_count];
                if (open_cameras(cameras_params, ecams, cameras_select, device_info, *cam_count, *config_folder)) 
                {
                    manager_context->state = FetchGame::ManagerState_CAMERAOPENED;
                } else {
                    manager_context->state = FetchGame::ManagerState_ERROR;
                }
                break;
            case FetchGame::ManagerState_STARTCAMTHREAD:
                if (start_camera_thread(camera_threads, cameras_params, ecams, camera_control, cameras_select, device_info, *cam_count, ptp_params, recording_setup->record_folder, recording_setup->encoder_basic_setup))
                {
                    manager_context->state = FetchGame::ManagerState_THREADREADY;
                } else {
                    manager_context->state = FetchGame::ManagerState_ERROR;
                }
                break;                
        }

        if (ptp_params->network_set_stop_ptp && ptp_params->ptp_stop_reached) {
            ptp_params->network_set_stop_ptp = false;
            for (auto &t : camera_threads)
                t.join();
            
            for (int i = 0; i < *cam_count; i++)
            {
                camera_threads.pop_back();
            }


            for (int i = 0; i < *cam_count; i++)
            {
                ptp_sync_off(&ecams[i].camera);
            }
            ptp_params->ptp_global_time = 0;
            ptp_params->ptp_stop_time = 0;
            ptp_params->ptp_counter = 0;
            ptp_params->ptp_stop_counter = 0;
            ptp_params->network_sync = true;
            ptp_params->network_set_start_ptp = false;
            ptp_params->ptp_stop_reached = false;
            
            camera_control->sync_camera = false;

            for (int i = 0; i < *cam_count; i++)
            {
                destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size);
                delete[] ecams[i].evt_frame;
                check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera));
                close_camera(&ecams[i].camera);
            }
            delete[] ecams;
            manager_context->state = FetchGame::ManagerState_RECORDSTOPPED;
        }
        usleep(1000);
    }

}


int main(int argc, char *argv[])
{
    if (enet_initialize() != 0)
    {
        quit_process(true, "ENET failed to initialize!");
    }

    signal(SIGINT, interruptHandler);

    EnetContext client;
    if (enet_initialize(&client, 3333, 1))
    {
        printf("Network Initialized!\n");
    }

    f32 last_time = tick();
    f32 current_time = tick();

    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);
    std::cout << "available no of cameras: " << cam_count << std::endl;

    flatbuffers::FlatBufferBuilder* fb_builder = new flatbuffers::FlatBufferBuilder(1024);
    std::string config_folder;
    RecordingContext recording_setup;
    ManagerContext manager_context;
    PTPParams *ptp_params = new PTPParams{0, 0, 0, 0, true, false, false, false};

    std::thread* manager_thread = new std::thread(&create_camera_manager, &cam_count, &manager_context, unsorted_device_info, device_info, &config_folder, &recording_setup, ptp_params);
    
    while (!quit_server)
    {
        current_time = tick();
        // Handle All Incoming Packets and Send any enqued packets, does this need to be on another thread?
        service_network(&client, current_time - last_time, [&](const ENetEvent &evnt)
        {
            switch (evnt.type) 
            {
                //New connection request or an existing peer accepted our connection request
                case ENET_EVENT_TYPE_CONNECT:
                    {
                        if (manager_context.state == FetchGame::ManagerState_IDLE) {
                            printf("Network: Successfully connected! Rescaning cameras. \n");
                            manager_context.state = FetchGame::ManagerState_CONNECT; // rescan number of cams
                        } else {
                            printf("Network: Successfully connected! \n");
                            client_send_bringup_message(&client, fb_builder, evnt.peer, cam_count, manager_context.state);
                        }
                    }
                    break;
                //Server has sent us a new packet
                case ENET_EVENT_TYPE_RECEIVE:
                    {
                        printf ("\n A packet of length %u was received from %s on channel %u.\n",
                            evnt.packet -> dataLength,
                            evnt.peer -> data,
                            evnt.channelID);

                        uint8_t* buffer_pointer = evnt.packet->data;
                        auto server_control = FetchGame::GetServer(buffer_pointer);
                        auto server_signal = server_control->control();

                        if (server_signal == FetchGame::ServerControl_OPENCAMERA) {
                            config_folder = server_control->config_folder()->c_str();
                            manager_context.state = FetchGame::ManagerState_OPENCAMERA;
                        }
                        else if (server_signal == FetchGame::ServerControl_STARTTHREAD)
                        {
                            recording_setup.record_folder = server_control->record_folder()->c_str();
                            recording_setup.encoder_basic_setup = server_control->encoder_setup()->c_str();
                            manager_context.state = FetchGame::ManagerState_STARTCAMTHREAD;
                        } else if (server_signal == FetchGame::ServerControl_QUIT) {
                            printf("Exit \n");
                            quit_server = true;
                        } else if (server_signal == FetchGame::ServerControl_STARTRECORDING) {
                            ptp_params->ptp_global_time = server_control->ptp_global_time();
                            std::cout << ptp_params->ptp_global_time << std::endl;
                            ptp_params->network_set_start_ptp = true;
                            manager_context.state = FetchGame::ManagerState_WAITSTOP;
                            client_send_state_update_message(&client, fb_builder, evnt.peer, manager_context.state);
                        } else if (server_signal == FetchGame::ServerControl_STOPRECORDING) {
                            // stop recording
                            printf("stop signal\n");
                            std::cout << server_control->ptp_global_time() << std::endl;
                            ptp_params->ptp_stop_time = server_control->ptp_global_time();
                            std::cout << ptp_params->ptp_stop_time << std::endl;
                            ptp_params->network_set_stop_ptp = true;
                        }
                        enet_packet_destroy(evnt.packet);
                    }
                    break;

                //Server has disconnected
                case ENET_EVENT_TYPE_DISCONNECT:
                    printf("Network: Server has disconnected!\n");
                    break;
            } });

        // coordinate with other thread
        if (manager_context.state == FetchGame::ManagerState_CONNECTED)
        {
            manager_context.state == FetchGame::ManagerState_IDLE;
            client_send_bringup_message(&client, fb_builder, &client.m_pNetwork->peers[0], cam_count, manager_context.state);
        }
        if (manager_context.state == FetchGame::ManagerState_CAMERAOPENED) {
            manager_context.state = FetchGame::ManagerState_WAITTHREAD;
            client_send_state_update_message(&client, fb_builder, &client.m_pNetwork->peers[0], manager_context.state);
        } else if (manager_context.state == FetchGame::ManagerState_THREADREADY)
        {
            manager_context.state = FetchGame::ManagerState_WAITSTART;
            client_send_state_update_message(&client, fb_builder, &client.m_pNetwork->peers[0], manager_context.state);
        } else if (manager_context.state == FetchGame::ManagerState_RECORDSTOPPED)
        {
            manager_context.state = FetchGame::ManagerState_IDLE;
            client_send_state_update_message(&client, fb_builder, &client.m_pNetwork->peers[0], manager_context.state);
        }
    
        usleep(1000);
        last_time = current_time;
    }

    manager_context.quit = true;
    manager_thread->join();

    // Disconnect
    enet_peer_disconnect(&client.m_pNetwork->peers[0], 0);
    uint8_t disconnected = false;
    /* Allow up to 3 seconds for the disconnect to succeed
     * and drop any packets received packets.
     */
    ENetEvent evnt;
    while (enet_host_service(client.m_pNetwork, &evnt, 3000) > 0)
    {
        switch (evnt.type)
        {
        case ENET_EVENT_TYPE_RECEIVE:
            enet_packet_destroy(evnt.packet);
            break;
        case ENET_EVENT_TYPE_DISCONNECT:
            puts("Disconnection succeeded.");
            disconnected = true;
            break;
        }
    }
    

    // Drop connection, since disconnection didn't successed
    if (!disconnected)
    {
        enet_peer_reset(&client.m_pNetwork->peers[0]);
    }
    enet_host_destroy(client.m_pNetwork);
    enet_deinitialize();
    return 0;
}
