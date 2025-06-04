// enet_thread.h
// ... other includes ...
#include "yolo_worker.h" // For YOLOv8Worker
#include <vector>        // For std::vector

// Assume yolo_workers and external_data_consumer_peer are accessible
// If they are global in orange.cpp, declare them as extern here.
// This is often not the cleanest approach for larger projects, but matches the current described setup.
extern std::vector<YOLOv8Worker*> yolo_workers;
extern ENetPeer* external_data_consumer_peer; // To store the specific peer

// ... (existing enet_thread.h content) ...

inline void create_enet_thread(EnetContext* server_enet_context, // Renamed 'server' to avoid conflict if yolo_workers comes from global
                              ConnectedServer* my_servers, 
                              INDIGOSignalBuilder* indigo_signal_builder, 
                              bool* quit_enet)
{
    while(!(*quit_enet)) {
        service_network(server_enet_context, ImGui::GetIO().DeltaTime, [&](const ENetEvent& evnt)
        {
            switch (evnt.type)
            {
            case ENET_EVENT_TYPE_CONNECT:
                printf ("A new client connected from %x:%u.\n", evnt.peer -> address.host, evnt.peer -> address.port);
                // TODO: Add logic here to identify if evnt.peer is the external data consumer
                // For now, let's assume any new connection that isn't Indigo or another Orange server could be it.
                // This identification logic needs to be robust.
                // Example: check against known IPs of `my_servers` and `indigo_signal_builder->indigo_connection`
                {
                    bool is_known_server = false;
                    for(int i=0; i<2; ++i) { // Assuming 2 my_servers
                        if (my_servers[i].peer == evnt.peer) {
                            is_known_server = true;
                            break;
                        }
                    }
                    if (indigo_signal_builder->indigo_connection == evnt.peer) {
                        is_known_server = true;
                    }

                    // If it's not a known server peer, assume it's the external data consumer for YOLO
                    // This is a simplified assumption.
                    if (!is_known_server && external_data_consumer_peer == nullptr) {
                        external_data_consumer_peer = evnt.peer;
                        std::cout << "External data consumer (potential YOLO ENet target) connected." << std::endl;
                        for (YOLOv8Worker* worker : yolo_workers) {
                            if (worker) {
                                worker->SetENetTarget(server_enet_context, external_data_consumer_peer);
                            }
                        }
                    }
                }
                break;

            case ENET_EVENT_TYPE_RECEIVE:
                {
                    // ... (existing ENET_EVENT_TYPE_RECEIVE logic) ...
                    uint8_t* buffer_pointer = evnt.packet->data;
                    auto server_control = FetchGame::GetServer(buffer_pointer);
                    
                    if (server_control->signal_type() == FetchGame::SignalType_ClientBringup) {
                        for (int i = 0; i < 2; i++) { // Assuming 2 my_servers
                            if (my_servers[i].peer == evnt.peer) {
                            }                           
                        }                        
                    } else if (server_control->signal_type() == FetchGame::SignalType_INDIGO) {
                        indigo_signal_builder->indigo_connection = evnt.peer;
                         // If Indigo connects, it's not the YOLO data consumer
                        if (external_data_consumer_peer == evnt.peer) {
                            external_data_consumer_peer = nullptr; // Clear it if Indigo was mistakenly assigned
                            for (YOLOv8Worker* worker : yolo_workers) {
                                if (worker) worker->SetENetTarget(server_enet_context, nullptr);
                            }
                        }
                    } // ... (other existing else if conditions) ...
                    enet_packet_destroy(evnt.packet);
                }
                break;

            case ENET_EVENT_TYPE_DISCONNECT:
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
                // Also update my_servers if the disconnected peer was one of them
                for(int i=0; i<2; ++i) {
                    if (my_servers[i].peer == evnt.peer) {
                        my_servers[i].peer = nullptr;
                        my_servers[i].connected = false;
                        my_servers[i].server_state = FetchGame::ManagerState_IDLE; // Or some disconnected state
                        my_servers[i].num_cameras = 0;
                        break;
                    }
                }
                if (indigo_signal_builder->indigo_connection == evnt.peer) {
                    indigo_signal_builder->indigo_connection = nullptr;
                }
                break;
            }
        });
        usleep(10);
    }
}