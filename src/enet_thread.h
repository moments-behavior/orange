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
                    
                    // CRITICAL: Only try obj_msg detection if it's NOT from a camera client
                    // Camera clients (dosa0/dosa1) only send Server messages, never obj_msg
                    // obj_msg should only come from INDIGO/CBOT connection
                    // If it's from a camera client, it's definitely a Server message
                    if (!is_camera_client) {
                        // Try to detect obj_msg by attempting to parse it
                        // obj_msg has cylinder1 and cylinder2 fields that Server messages don't have
                        try {
                            auto obj_msg = Obj::Getobj_msg(buffer_pointer);
                            if (obj_msg) {
                                // Try to access obj_msg-specific fields
                                // For obj_msg, these should be accessible (even if null)
                                // For Server messages parsed as obj_msg, accessing these might give garbage
                                try {
                                    auto c1 = obj_msg->cylinder1();
                                    auto c2 = obj_msg->cylinder2();
                                    
                                    // Additional verification: try to parse as Server and check if it gives invalid data
                                    // If Server parsing gives invalid signal_type, it's likely obj_msg
                                    bool likely_obj_msg = false;
                                    try {
                                        auto test_server = FetchGame::GetServer(buffer_pointer);
                                        if (test_server) {
                                            auto test_signal = test_server->signal_type();
                                            // If signal_type is invalid, it's probably obj_msg misparsed as Server
                                            if (::flatbuffers::IsOutRange(test_signal, FetchGame::SignalType_ClientBringup, FetchGame::SignalType_CalibrationDone)) {
                                                likely_obj_msg = true;
                                            }
                                        }
                                    } catch (...) {
                                        // Server parsing failed - likely obj_msg
                                        likely_obj_msg = true;
                                    }
                                    
                                    // If we can access cylinder fields AND Server parsing is invalid, it's obj_msg
                                    if (likely_obj_msg) {
                                        is_obj_msg = true;
                                        std::cout << "DEBUG: Detected obj_msg from non-camera-client (Server parsing invalid), skipping Server parsing" << std::endl;
                                    }
                                } catch (...) {
                                    // Access failed, might not be obj_msg
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
                                        // Only update server state if this is from a camera client
                                        bool updated_camera_client = false;
                                        for (int i = 0; i < 2; i++) {
                                            if (my_servers[i].peer == evnt.peer) {
                                                auto server_name = server_control->server_mesg()->server_name()->c_str();
                                                auto server_num_cameras = server_control->server_mesg()->num_cameras();
                                                auto server_state = server_control->server_state();
                                                my_servers[i].num_cameras = server_num_cameras;
                                                my_servers[i].server_state = server_state;
                                                updated_camera_client = true;
                                                std::cout << "DEBUG: Updated camera client " << i << " state to " << (int)server_state << " from ClientBringup" << std::endl;
                                            }                           
                                        }
                                        
                                        // Also set this as potential CBOT connection for OBB messages
                                        // But only if it's not a camera client (CBOT sends ClientBringup too)
                                        if (!updated_camera_client) {
                                            std::cout << "DEBUG: Setting CBOT connection from ClientBringup signal (non-camera-client)" << std::endl;
                                            indigo_signal_builder->indigo_connection = evnt.peer;
                                        } else {
                                            std::cout << "DEBUG: ClientBringup from camera client, also setting as CBOT connection" << std::endl;
                                            indigo_signal_builder->indigo_connection = evnt.peer;
                                        }
                                        
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
                                    else if (signal_type == FetchGame::SignalType_ClientStateUpdate) {
                                        // Only update server state if this is from a camera client
                                        // AND verify it's a valid Server message by checking it has server_state field
                                        if (is_camera_client) {
                                            try {
                                                auto server_state = server_control->server_state();
                                                // Verify server_state is valid before updating
                                                if (!::flatbuffers::IsOutRange(server_state, FetchGame::ManagerState_IDLE, FetchGame::ManagerState_WAITSTOP)) {
                                                    for (int i = 0; i < 2; i++) {
                                                        if (my_servers[i].peer == evnt.peer) {
                                                            auto old_state = my_servers[i].server_state;
                                                            my_servers[i].server_state = server_state;
                                                            std::cout << "DEBUG: Updated camera client " << i << " state from " << (int)old_state << " to " << (int)server_state << " from ClientStateUpdate" << std::endl;
                                                        }
                                                    }
                                                } else {
                                                    std::cout << "DEBUG: Invalid server_state: " << (int)server_state << ", ignoring ClientStateUpdate" << std::endl;
                                                }
                                            } catch (...) {
                                                std::cout << "DEBUG: Failed to access server_state, ignoring ClientStateUpdate message" << std::endl;
                                            }
                                        } else {
                                            std::cout << "DEBUG: ClientStateUpdate from non-camera-client peer, ignoring (this should never update server state)" << std::endl;
                                        }
                                    }
                                    else {
                                        // Other signal types - don't update server state
                                        std::cout << "DEBUG: Received signal type " << (int)signal_type << ", no state update needed" << std::endl;
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
