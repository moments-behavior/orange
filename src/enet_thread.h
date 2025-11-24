#include "project.h"
#include "network_base.h"
#include "imgui.h"
#include "global.h"
#include "obj_generated.h"
#include "camera.h"

void create_enet_thread(EnetContext* server, ConnectedServer* my_servers, INDIGOSignalBuilder* indigo_signal_builder, bool* quit_enet, PTPParams* ptp_params)
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
                    bool is_camera_client = false;
                    
                    // Check if this is from a known camera client (dosa0/dosa1)
                    for (int i = 0; i < 2; i++) {
                        if (my_servers[i].peer == evnt.peer) {
                            is_camera_client = true;
                            break;
                        }
                    }
                    
                    // CRITICAL INSIGHT: We only SEND obj_msg to CBOT, we never RECEIVE them back
                    // So obj_msg should never be in the receive path. The real issue must be something else.
                    // 
                    // The safest approach: Only process Server messages from known camera clients
                    // If it's not from a camera client, try to parse as Server but be very careful
                    // about state updates - only update from camera clients.
                    
                    // Try parsing as Server message
                    {
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
                                                    // CRITICAL SAFETY CHECK: Prevent accepting IDLE state from camera clients during active recording
                                                    // This prevents the button from disappearing when clients incorrectly send IDLE during recording
                                                    if (server_state == FetchGame::ManagerState_IDLE && 
                                                        ptp_params && 
                                                        ptp_params->ptp_start_reached && 
                                                        !ptp_params->ptp_stop_reached) {
                                                        std::cout << "DEBUG SERVER: REJECTING IDLE state update from camera client - actively recording! "
                                                                  << "ptp_start_reached=" << ptp_params->ptp_start_reached 
                                                                  << ", ptp_stop_reached=" << ptp_params->ptp_stop_reached 
                                                                  << ". Ignoring ClientStateUpdate." << std::endl;
                                                    } else {
                                                        for (int i = 0; i < 2; i++) {
                                                            if (my_servers[i].peer == evnt.peer) {
                                                                auto old_state = my_servers[i].server_state;
                                                                my_servers[i].server_state = server_state;
                                                                std::cout << "DEBUG: Updated camera client " << i << " state from " << (int)old_state << " to " << (int)server_state << " from ClientStateUpdate" << std::endl;
                                                            }
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
                            // If parsing as Server fails, it might be obj_msg or some other message type
                            // Since we only send obj_msg and never receive them, this is unexpected
                            // Log it but don't process it
                            std::cout << "DEBUG: Failed to parse message as Server (might be obj_msg or unknown type), ignoring" << std::endl;
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
