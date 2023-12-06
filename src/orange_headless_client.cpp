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

#define evt_buffer_size 100

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

    cameras_params = new CameraParams[num_cameras];
    cameras_select = new CameraEachSelect[num_cameras];

    for (int i = 0; i < num_cameras; i++)
    {
        set_camera_params(&cameras_params[i], &device_info[i], camera_config_files, i, num_cameras);
        open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
    }
    return true;
}


bool start_camera_thread(std::vector<std::thread> &camera_threads, CameraParams *cameras_params, CameraEmergent *ecams, CameraControl *camera_control, CameraEachSelect *cameras_select, GigEVisionDeviceInfo *device_info, int num_cameras, PTPParams *ptp_params, std::string record_folder)
{
    std::cout << "start camera sthread..." << std::endl;
    allocate_camera_frame_buffers(ecams, cameras_params, evt_buffer_size, num_cameras);
    camera_control->record_video = true;
    camera_control->subscribe = true;
    camera_control->sync_camera = true;
    std::string encoder_setup = "-codec h264 -preset p1 -fps " + std::to_string(cameras_params[0].frame_rate);

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
        camera_threads.push_back(std::thread(&aquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, nullptr, encoder_setup, record_folder, ptp_params));
    }

    return true;
}

int main(int argc, char *argv[])
{
    if (enet_initialize() != 0)
    {
        quit_process(true, "ENET failed to initialize!");
    }

    ENetPeer *server_connection;
    EnetContext client;
    if (enet_initialize(&client, 0, 1))
    {
        printf("Network Initialized!\n");
        server_connection = connect_peer(&client, 192, 168, 20, 10, 3333);
        // server_connection = connect_peer(&client, 127, 0, 0, 1, 3333);
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
    CameraEachSelect *cameras_select;

    ecams = new CameraEmergent[cam_count];
    CameraControl *camera_control = new CameraControl;
    PTPParams *ptp_params = new PTPParams{0, 0, 0, 0, true, false, false, false};
    
    flatbuffers::FlatBufferBuilder builder(1024);

    bool quit_server = false;

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
                        if (evnt.peer == server_connection)
                        {
                            printf("Network: Successfully connected to server! \n");
                            client_send_bringup_message(&client, builder, server_connection, cam_count);
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

                        if (server_signal == FetchGame::ServerControl_OPEN) {
                            std::string config_folder = server_control->config_folder()->c_str();
                            if (open_cameras(cameras_params, ecams, cameras_select, device_info, cam_count, config_folder)) {
                                client_send_camera_open_message(&client, builder, server_connection);
                            }
                        }
                        else if (server_signal == FetchGame::ServerControl_START)
                        {
                            std::string record_folder = server_control->record_folder()->c_str();
                            if(start_camera_thread(camera_threads, cameras_params, ecams, camera_control, cameras_select, device_info, cam_count, ptp_params, record_folder)) 
                            {
                                client_send_thread_start_message(&client, builder, server_connection);
                            };
                        } else if (server_signal == FetchGame::ServerControl_QUIT) {
                            printf("Exit \n");
                            quit_server = true;
                        } else if (server_signal == FetchGame::ServerControl_SETPTP) {
                            ptp_params->ptp_global_time = server_control->ptp_global_time();
                            std::cout << ptp_params->ptp_global_time << std::endl;
                            ptp_params->servers_ready = true;
                            client_send_ptp_set_message(&client, builder, server_connection);
                        } else if (server_signal == FetchGame::ServerControl_STOP) {
                            // stop recording
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
                    {
                        printf("Network: Server has disconnected!");
                    }
                    break;
                } });

        if (ptp_params->network_set_stop_ptp && ptp_params->ptp_stop_reached) {
            ptp_params->network_set_stop_ptp = false;
            for (auto &t : camera_threads)
                t.join();
            
            for (int i = 0; i < cam_count; i++)
            {
                camera_threads.pop_back();
            }


            for (int i = 0; i < cam_count; i++)
            {
                ptp_sync_off(&ecams[i].camera);
            }
            ptp_params->ptp_global_time = 0;
            ptp_params->ptp_stop_time = 0;
            ptp_params->ptp_counter = 0;
            ptp_params->ptp_stop_counter = 0;
            ptp_params->network_sync = true;
            ptp_params->servers_ready = false;
            ptp_params->ptp_stop_reached = false;
            
            camera_control->sync_camera = false;

            for (int i = 0; i < cam_count; i++)
            {
                destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size);
                delete[] ecams[i].evt_frame;
                check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera));
                close_camera(&ecams[i].camera);
            }
            // send signal the thread is idle
            client_send_record_done_message(&client, builder, server_connection);
        }

        usleep(10000); // sleep for 10ms
        last_time = current_time;
    }

    // Disconnect
    enet_peer_disconnect(server_connection, 0);
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
        enet_peer_reset(server_connection);
    }
    enet_host_destroy(client.m_pNetwork);
    quit_process();
    return 0;
}
