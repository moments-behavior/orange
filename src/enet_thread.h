#include "project.h"
#include "network_base.h"
#include "imgui.h"
#include "global.h"
#include "obj_generated.h"

void create_enet_thread(EnetContext* server, ConnectedServer* my_servers, INDIGOSignalBuilder* indigo_signal_builder, bool* quit_enet)
{
    while(!(*quit_enet)) {
        service_network(server, ImGui::GetIO().DeltaTime, [&](const ENetEvent& evnt)
        {
            switch (evnt.type)
            {
            case ENET_EVENT_TYPE_CONNECT:
                printf ("A new client connected from %x:%u.\n", evnt.peer -> address.host, evnt.peer -> address.port);
                break;

            case ENET_EVENT_TYPE_RECEIVE:
                {
                    uint8_t* buffer_pointer = evnt.packet->data;
                    bool is_obj_msg = false;
                    
                    // Check if this is an obj_msg (OBB detection message) from INDIGO connection
                    // If so, skip Server message parsing to avoid corrupting server states
                    if (indigo_signal_builder->indigo_connection == evnt.peer && 
                        indigo_signal_builder->indigo_connection != nullptr) {
                        // Try to parse as obj_msg first
                        auto obj_msg = Obj::Getobj_msg(buffer_pointer);
                        if (obj_msg) {
                            // Check if this looks like a valid obj_msg by checking if it has the expected structure
                            // obj_msg should have cylinder1 and cylinder2 fields
                            if (obj_msg->cylinder1() && obj_msg->cylinder2()) {
                                // This is an OBB detection message, ignore it here
                                // (it's handled in opengldisplay.cpp)
                                is_obj_msg = true;
                            }
                        }
                    }
                    
                    // If it's an obj_msg, skip Server message parsing
                    if (!is_obj_msg) {
                        // Parse as Server message for camera client communication
                        auto server_control = FetchGame::GetServer(buffer_pointer);
                        std::cout << "DEBUG: Received signal type: " << (int)server_control->signal_type() << std::endl;
                        
                        if (server_control->signal_type() == FetchGame::SignalType_ClientBringup) {
                            // Handle general client bring-up
                            for (int i = 0; i < 2; i++) {
                                if (my_servers[i].peer == evnt.peer) {
                                    auto server_name = server_control->server_mesg()->server_name()->c_str();
                                    auto server_num_cameras = server_control->server_mesg()->num_cameras();
                                    auto server_state = server_control->server_state();
                                    my_servers[i].num_cameras = server_num_cameras;
                                    my_servers[i].server_state = server_state;
                                }                           
                            }
                            
                            // Also set this as potential CBOT connection for OBB messages
                            std::cout << "DEBUG: Setting CBOT connection from ClientBringup signal" << std::endl;
                            indigo_signal_builder->indigo_connection = evnt.peer;
                            
                        } else if (server_control->signal_type() == FetchGame::SignalType_INDIGO) {
                            std::cout << "DEBUG: Setting CBOT connection from INDIGO signal" << std::endl;
                            indigo_signal_builder->indigo_connection = evnt.peer;
                        } else if (server_control->signal_type() == FetchGame::SignalType_CalibrationPoseReached) {
                            std::cout << "From Indigo: Calibration pose reached." << std::endl;
                            calib_state = CalibPoseReached;
                        } else if (server_control->signal_type() == FetchGame::SignalType_CalibrationDone) {
                            std::cout << "From Indigo: Calibration done." << std::endl;
                            calib_state = CalibIdle;
                        }
                        else {
                            // Only update server state if this is from a camera client
                            // Skip if this is an obj_msg from INDIGO connection
                            for (int i = 0; i < 2; i++) {
                                if (my_servers[i].peer == evnt.peer) {
                                    auto server_state = server_control->server_state();
                                    my_servers[i].server_state = server_state;
                                }
                            }
                        }
                    }
                    enet_packet_destroy(evnt.packet);
                }
                break;

            case ENET_EVENT_TYPE_DISCONNECT:
                printf("- Client %d has disconnected.\n", evnt.peer->incomingPeerID);
                break;
            }
        });
        usleep(10);
    }
}
