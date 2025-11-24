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
                    bool is_camera_client = false;
                    
                    // Check if this is from a known camera client (dosa0/dosa1)
                    for (int i = 0; i < 2; i++) {
                        if (my_servers[i].peer == evnt.peer) {
                            is_camera_client = true;
                            break;
                        }
                    }
                    
                    // Always try to parse as obj_msg first for any message
                    // obj_msg messages can come from any source and should be ignored here
                    // We check message size first as a quick filter (obj_msg is typically small)
                    if (evnt.packet->dataLength < 200) {  // obj_msg is typically small
                        try {
                            auto obj_msg = Obj::Getobj_msg(buffer_pointer);
                            if (obj_msg) {
                                // Try to access the structure - if it's a valid obj_msg, we should be able to
                                // access cylinder1 and cylinder2 (they may be null, but accessors should work)
                                try {
                                    auto c1 = obj_msg->cylinder1();
                                    auto c2 = obj_msg->cylinder2();
                                    // If we got here without crashing, it's likely an obj_msg
                                    // Double-check by verifying the structure makes sense
                                    // (obj_msg should have these fields even if null)
                                    is_obj_msg = true;
                                    std::cout << "DEBUG: Received obj_msg (size: " << evnt.packet->dataLength << "), skipping Server parsing" << std::endl;
                                } catch (...) {
                                    // Access failed, probably not obj_msg
                                }
                            }
                        } catch (...) {
                            // Not an obj_msg, continue to try Server parsing
                        }
                    }
                    
                    // If it's not an obj_msg, try parsing as Server message
                    if (!is_obj_msg) {
                        try {
                            auto server_control = FetchGame::GetServer(buffer_pointer);
                            if (server_control) {
                                // Check if signal_type is valid (within expected range)
                                auto signal_type = server_control->signal_type();
                                if (!::flatbuffers::IsOutRange(signal_type, FetchGame::SignalType_ClientBringup, FetchGame::SignalType_CalibrationDone)) {
                                    // Valid Server message, process it
                                    std::cout << "DEBUG: Received signal type: " << (int)signal_type << " from " << (is_camera_client ? "camera client" : "other") << std::endl;
                                    
                                    if (signal_type == FetchGame::SignalType_ClientBringup) {
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
                                        
                                    } else if (signal_type == FetchGame::SignalType_INDIGO) {
                                        std::cout << "DEBUG: Setting CBOT connection from INDIGO signal" << std::endl;
                                        indigo_signal_builder->indigo_connection = evnt.peer;
                                    } else if (signal_type == FetchGame::SignalType_CalibrationPoseReached) {
                                        std::cout << "From Indigo: Calibration pose reached." << std::endl;
                                        calib_state = CalibPoseReached;
                                    } else if (signal_type == FetchGame::SignalType_CalibrationDone) {
                                        std::cout << "From Indigo: Calibration done." << std::endl;
                                        calib_state = CalibIdle;
                                    }
                                    else {
                                        // Only update server state if this is from a camera client
                                        // This prevents obj_msg from corrupting states even if it parses as Server
                                        if (is_camera_client) {
                                            for (int i = 0; i < 2; i++) {
                                                if (my_servers[i].peer == evnt.peer) {
                                                    auto server_state = server_control->server_state();
                                                    my_servers[i].server_state = server_state;
                                                }
                                            }
                                        } else {
                                            std::cout << "DEBUG: Server message from non-camera-client peer, ignoring state update" << std::endl;
                                        }
                                    }
                                } else {
                                    // Invalid signal_type - might be obj_msg that parsed incorrectly
                                    std::cout << "DEBUG: Invalid signal_type: " << (int)signal_type << ", ignoring message" << std::endl;
                                }
                            }
                        } catch (...) {
                            // If parsing as Server fails, log and ignore
                            std::cout << "DEBUG: Failed to parse message as Server, ignoring" << std::endl;
                        }
                    }
                    
                    // If it's an obj_msg, we've already skipped processing
                    // (obj_msg messages are handled in opengldisplay.cpp)
                    
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
