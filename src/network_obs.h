#ifndef ORANGE_NETWORK
#define ORANGE_NETWORK
#include "fetch_generated.h"
#include <enet/enet.h>
#include <functional>
#include <iostream>

enum PacketTransportType {
    // Packet will be sent, but may be lost in transit
    //  - Best for regular updates that are time-sensitive e.g. position updates
    //  - See UDP Protocol
    PACKET_TRANSPORT_UNRELABLE = 0,

    // Packet will be sent and tracked, resending after timeout if peer has not
    // recieved it yet
    //  - Best for data that MUST reach the client, no matter how long (or how
    //  many attempts) it takes e.g. game start/end and player deaths
    //  - See TCP Protocol
    PACKET_TRANSPORT_RELIABLE = ENET_PACKET_FLAG_RELIABLE
};

struct EnetContext {
    ENetHost *m_pNetwork = NULL;
    float m_IncomingKb = 0.0f; /// Usually refered to as Rx	(Recieve rate)
    float m_OutgoingKb = 0.0f; /// Usually refered to as Tx (Transmit rate)
    float m_SecondTimer = 0.0f;
};

bool enet_initialize(EnetContext *enet_context, uint16_t external_port_number,
                     size_t max_peers);
void enet_release(EnetContext *enet_context);
ENetPeer *connect_peer(EnetContext *enet_context, uint8_t ip_part1,
                       uint8_t ip_part2, uint8_t ip_part3, uint8_t ip_part4,
                       uint16_t port_number);
void enqueue_packet(EnetContext *enet_context, ENetPeer *peer,
                    PacketTransportType transport_type, void *packet_data,
                    size_t data_length);
void service_network(EnetContext *enet_context, float dt,
                     std::function<void(const ENetEvent &)> callback);
void send_indigo_message(EnetContext *enet_context,
                         flatbuffers::FlatBufferBuilder *builder,
                         ENetPeer *indigo_connection,
                         FetchGame::SignalType signal_type);
#endif
