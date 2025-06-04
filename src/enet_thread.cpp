// src/enet_thread.cpp
#include "enet_thread.h"
#include "imgui.h" // For ImGui::GetIO().DeltaTime
#include <unistd.h> // For usleep


// The actual definition of the function
void create_enet_thread(EnetContext* server_enet_context,
                        ConnectedServer* my_servers,
                        INDIGOSignalBuilder* indigo_signal_builder,
                        bool* quit_enet)
{
    std::cout << "enet_thread.cpp: Address of EnetContext 'server_enet_context' received: " << static_cast<void*>(server_enet_context) << std::endl;

    while(!(*quit_enet)) {
        service_network(server_enet_context, ImGui::GetIO().DeltaTime, [&](const ENetEvent& evnt)
        {
            switch (evnt.type)
            {
            case ENET_EVENT_TYPE_CONNECT:
                {
                    printf ("A new client connected from %x:%u.\n", evnt.peer -> address.host, evnt.peer -> address.port);
                    bool is_known_server = false;
                    for(int i=0; i<2; ++i) {
                        if (my_servers[i].peer == evnt.peer) {
                            is_known_server = true;
                            break;
                        }
                    }
                    if (indigo_signal_builder->indigo_connection == evnt.peer) {
                        is_known_server = true;
                    }

                    if (!is_known_server && external_data_consumer_peer == nullptr) {
                        external_data_consumer_peer = evnt.peer;
                        std::cout << "External data consumer (potential YOLO ENet target) connected. Peer ID: " << external_data_consumer_peer->incomingPeerID << std::endl;
                        std::cout << "enet_thread.cpp: Attempting to set target for YOLO workers. Current yolo_workers.size() = " << yolo_workers.size()
                                  << ". server_enet_context address: " << static_cast<void*>(server_enet_context) << std::endl;

                        for (YOLOv8Worker* worker : yolo_workers) {
                            if (worker) {
                                std::cout << "enet_thread.cpp: Calling SetENetTarget for worker at " << static_cast<void*>(worker)
                                          << " with server_enet_context: " << static_cast<void*>(server_enet_context)
                                          << " and peer: " << static_cast<void*>(external_data_consumer_peer) << std::endl;
                                worker->SetENetTarget(server_enet_context, external_data_consumer_peer);
                            } else {
                                std::cout << "enet_thread.cpp: Encountered nullptr worker in yolo_workers vector." << std::endl;
                            }
                        }
                    }
                }
                break;

            case ENET_EVENT_TYPE_RECEIVE:
                {
                    uint8_t* buffer_pointer = evnt.packet->data;
                    flatbuffers::Verifier verifier(buffer_pointer, evnt.packet->dataLength);

                    if (!flatbuffers::BufferHasIdentifier(buffer_pointer, Orange::Network::RootMessageIdentifier())) {
                        std::cerr << "Received packet with invalid/unknown identifier." << std::endl;
                        enet_packet_destroy(evnt.packet);
                        break;
                    }

                    if (!Orange::Network::VerifyRootMessageBuffer(verifier)) {
                        std::cerr << "Received invalid RootMessage buffer." << std::endl;
                        enet_packet_destroy(evnt.packet);
                        break;
                    }

                    auto root_message = Orange::Network::GetRootMessage(buffer_pointer);
                    Orange::Network::Payload payload_type_val = root_message->payload_type();

                    if (payload_type_val == Orange::Network::Payload::Payload_FetchGame_Server) {
                        auto server_control = static_cast<const FetchGame::Server*>(root_message->payload());
                        if (server_control) {
                            if (server_control->signal_type() == FetchGame::SignalType_ClientBringup) {
                                for (int i = 0; i < 2; i++) {
                                    if (my_servers[i].peer == evnt.peer) {
                                        my_servers[i].num_cameras = server_control->server_mesg()->num_cameras();
                                        my_servers[i].server_state = server_control->server_state();
                                    }
                                }
                            } else if (server_control->signal_type() == FetchGame::SignalType_ClientStateUpdate) {
                                for (int i = 0; i < 2; i++) {
                                    if (my_servers[i].peer == evnt.peer) {
                                        my_servers[i].server_state = server_control->server_state();
                                    }
                                }
                            }
                            else if (server_control->signal_type() == FetchGame::SignalType_INDIGO) {
                                indigo_signal_builder->indigo_connection = evnt.peer;
                                if (external_data_consumer_peer == evnt.peer) {
                                    external_data_consumer_peer = nullptr;
                                    for (YOLOv8Worker* worker : yolo_workers) {
                                        if (worker) worker->SetENetTarget(server_enet_context, nullptr);
                                    }
                                }
                            } else if (server_control->signal_type() == FetchGame::SignalType_CalibrationPoseReached) {
                                std::cout << "Received CalibrationPoseReached from INDIGO" << std::endl;
                                calib_state.store(CalibPoseReached);
                            } else if (server_control->signal_type() == FetchGame::SignalType_CalibrationDone) {
                                std::cout << "Received CalibrationDone from INDIGO" << std::endl;
                                calib_state.store(CalibIdle);
                            }
                        }
                    } else if (payload_type_val == Orange::Network::Payload::Payload_Orange_VisionData_YoloFrameDetections) {
                        auto yolo_data = static_cast<const Orange::VisionData::YoloFrameDetections*>(root_message->payload());
                        if (yolo_data) {
                            std::cout << "Received External BoundingBox Data for camera: " << (yolo_data->camera_serial() ? yolo_data->camera_serial()->str() : "N/A") << std::endl;
                            std::cout << "  Timestamp: " << yolo_data->timestamp() << ", Frame ID: " << yolo_data->frame_id() << std::endl;
                            if (yolo_data->detections()) {
                                for (const auto* det : *(yolo_data->detections())) {
                                    if (det->box()) {
                                        std::cout << "    Box: x=" << det->box()->x() << ", y=" << det->box()->y()
                                                  << ", w=" << det->box()->width() << ", h=" << det->box()->height()
                                                  << ", label=" << det->box()->label() << ", prob=" << det->box()->probability() << std::endl;
                                    }
                                }
                            }
                        }
                    } else {
                        std::cout << "Received RootMessage with unknown payload type." << std::endl;
                    }
                    enet_packet_destroy(evnt.packet);
                }
                break;

            case ENET_EVENT_TYPE_DISCONNECT:
                {
                    printf("- Client %d has disconnected.\n", evnt.peer->incomingPeerID);
                    if (evnt.peer == external_data_consumer_peer) {
                        std::cout << "External data consumer (YOLO ENet target) disconnected." << std::endl;
                        external_data_consumer_peer = nullptr;
                        for (YOLOv8Worker* worker : yolo_workers) {
                            if (worker) {
                                worker->SetENetTarget(server_enet_context, nullptr);
                            }
                        }
                    }
                    for(int i=0; i<2; ++i) {
                        if (my_servers[i].peer == evnt.peer) {
                            my_servers[i].peer = nullptr;
                            my_servers[i].connected = false;
                            my_servers[i].server_state = FetchGame::ManagerState_IDLE;
                            my_servers[i].num_cameras = 0;
                            break;
                        }
                    }
                    if (indigo_signal_builder->indigo_connection == evnt.peer) {
                        indigo_signal_builder->indigo_connection = nullptr;
                    }
                }
                break;
            default:
                break;
            }
        });
        usleep(10);
    }
}