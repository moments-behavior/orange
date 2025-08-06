#include "global.h"
#include "imgui.h"
#include "network_base.h"
#include "project.h"

void create_enet_thread(EnetContext *server, ConnectedServer *my_servers,
                        bool *quit_enet) {
    while (!(*quit_enet)) {
        service_network(
            server, ImGui::GetIO().DeltaTime, [&](const ENetEvent &evnt) {
                switch (evnt.type) {
                case ENET_EVENT_TYPE_CONNECT:
                    printf("A new client connected from %x:%u.\n",
                           evnt.peer->address.host, evnt.peer->address.port);
                    break;

                case ENET_EVENT_TYPE_RECEIVE: {
                    uint8_t *buffer_pointer = evnt.packet->data;
                    auto server_control = FetchGame::GetServer(buffer_pointer);

                    if (server_control->signal_type() ==
                        FetchGame::SignalType_ClientBringup) {
                        for (int i = 0; i < 2; i++) {
                            if (my_servers[i].peer == evnt.peer) {
                                auto server_name = server_control->server_mesg()
                                                       ->server_name()
                                                       ->c_str();
                                auto server_num_cameras =
                                    server_control->server_mesg()
                                        ->num_cameras();
                                auto server_state =
                                    server_control->server_state();
                                my_servers[i].num_cameras = server_num_cameras;
                                my_servers[i].server_state = server_state;
                            }
                        }
                    } else if (server_control->signal_type() ==
                               FetchGame::SignalType_INDIGO) {
                        indigo_signal_builder.indigo_connection = evnt.peer;
                    } else if (server_control->signal_type() ==
                               FetchGame::SignalType_CalibrationPoseReached) {
                        std::cout << "From Indigo: Calibration pose reached."
                                  << std::endl;
                        calib_state = CalibPoseReached;
                    } else if (server_control->signal_type() ==
                               FetchGame::SignalType_CalibrationDone) {
                        std::cout << "From Indigo: Calibration done."
                                  << std::endl;
                        calib_state = CalibIdle;
                    } else {
                        for (int i = 0; i < 2; i++) {
                            if (my_servers[i].peer == evnt.peer) {
                                auto server_state =
                                    server_control->server_state();
                                my_servers[i].server_state = server_state;
                            }
                        }
                    }
                    enet_packet_destroy(evnt.packet);
                } break;

                case ENET_EVENT_TYPE_DISCONNECT:
                    printf("- Client %d has disconnected.\n",
                           evnt.peer->incomingPeerID);
                    break;
                }
            });
        usleep(10);
    }
}
