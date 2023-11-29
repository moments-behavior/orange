#include "network_base.h"

bool enet_initialize(EnetContext* enet_context, uint16_t external_port_number, size_t max_peers)
{
    ENetAddress address;
    address.host = ENET_HOST_ANY;
    address.port = external_port_number;

    enet_context->m_pNetwork = enet_host_create(
        (external_port_number == 0) ? NULL : &address,	//the address at which other peers may connect to this host. If NULL, then no peers may connect to the host.
        max_peers,												//the maximum number of peers that should be allocated for the host.
        1,												//the maximum number of channels allowed; if 0, then this is equivalent to ENET_PROTOCOL_MAXIMUM_CHANNEL_COUNT
        0,												//downstream bandwidth of the host in bytes/second; if 0, ENet will assume unlimited bandwidth.
        0);												//upstream bandwidth of the host in bytes/second; if 0, ENet will assume unlimited bandwidth.

    if (enet_context->m_pNetwork == NULL)
    {
        printf("Unable to initialize Network Host!");
        return false;
    }

    return true;
}


void enet_release(EnetContext* enet_context)
{
    if (enet_context->m_pNetwork != NULL)
    {
        enet_host_destroy(enet_context->m_pNetwork);
        enet_context->m_pNetwork = NULL;

        enet_context->m_IncomingKb = 0.0f;
        enet_context->m_OutgoingKb = 0.0f;
        enet_context->m_SecondTimer = 0.0f;
    }
}


// Attempt to connect to a peer with a given IP4 Addr:Port No
// - Example usage: BeginConnect(127,0,0,1, 1234) to connect to localhost on port 1234
// - Note: ENetPeer pointer is used to identify the peer and is needed to send/recieve packets to that computer
ENetPeer* connect_peer(EnetContext* enet_context, uint8_t ip_part1, uint8_t ip_part2, uint8_t ip_part3, uint8_t ip_part4, uint16_t port_number)
{
    if (enet_context->m_pNetwork != NULL)
    {
        ENetAddress address;
        address.port = port_number;

        //Host IP4 address must be condensed into a 32 bit integer
        address.host = (ip_part4 << 24) | (ip_part3 << 16) | (ip_part2 << 8) | (ip_part1);

        ENetPeer* peer = enet_host_connect(enet_context->m_pNetwork, &address, 2, 0);
        if (peer == NULL)
        {
            printf("Unable to connect to peer: %d.%d.%d.%d:%d", ip_part1, ip_part2, ip_part3, ip_part4, port_number);
        }

        return peer;
    }
    else
    {
        printf("Unable to connect to peer: Network not initialized!");
        return NULL;
    }
}


// Enqueues data to be sent to peer computer over the network.
// - Note: All enqueued packets will automatically be sent the next time 'ServiceNetwork' is called
void enqueue_packet(EnetContext* enet_context, ENetPeer* peer, PacketTransportType transport_type, void* packet_data, size_t data_length)
{
    if (enet_context->m_pNetwork != NULL)
    {
        if (peer != NULL)
        {
            ENetPacket* packet = enet_packet_create(packet_data, data_length, transport_type);
            enet_peer_send(peer, 0, packet);
        }
        else
        {
            printf("Unable to enqueue packet: Peer not initialized!");
        }
    }
    else
    {
        printf("Unable to enqueue packet: Network not initialized!");
    }
}


// Locks thread and waits x milliseconds for a network event to trigger. 
// - Returns true if event recieved or false otherwise
// - Events include:
//			ENET_EVENT_TYPE_CONNECT, returned when a peer is connected and has responded via 'BeginConnect' or an external peer is attempting to connect to us and awaiting our confirmation
//			ENET_EVENT_TYPE_RECEIVE, returned when a packet from a peer has been recieved and is awaiting processing
//			ENET_EVENT_TYPE_DISCONNECT, returned when a peer is disconnected
void service_network(EnetContext* enet_context, float dt, std::function<void(const ENetEvent&)> callback)
{
    if (enet_context->m_pNetwork != NULL)
    {
        //Handle all incoming packets & send any packets awaiting dispatch
        ENetEvent event;
        while (enet_host_service(enet_context->m_pNetwork, &event, 0) > 0)
        {
            callback(event);
        }


        //Update Transmit / Recieve bytes per second
        enet_context->m_SecondTimer += dt;
        if (enet_context->m_SecondTimer >= 1.0f)
        {
            enet_context->m_SecondTimer = 0.0f;

            enet_context->m_IncomingKb = float(enet_context->m_pNetwork->totalReceivedData / 128.0); // - 8 bits in a byte and 1024 bits in a KiloBit
            enet_context->m_OutgoingKb = float(enet_context->m_pNetwork->totalSentData / 128.0);
            enet_context->m_pNetwork->totalReceivedData = 0;
            enet_context->m_pNetwork->totalSentData = 0;
        }
    }
    else
    {
        printf("Unable to service network: Network not initialized!");
    }
}


void send_bringup_message(EnetContext* enet_context, flatbuffers::FlatBufferBuilder& builder, ENetPeer *server_connection, const char* hostname, int cam_count)
{
    builder.Clear();
    auto server_name = builder.CreateString(hostname);
    auto message_fb = FetchGame::Createbring_up_message(builder, server_name, cam_count);
    FetchGame::ServerBuilder server_builder(builder);
    server_builder.add_server_mesg(message_fb);
    auto server_fb = server_builder.Finish();
    builder.Finish(server_fb);
    uint8_t *server_buffer = builder.GetBufferPointer();
    int server_buf_size = builder.GetSize();
    ENetPacket* enet_packet = enet_packet_create(server_buffer, server_buf_size, 0);
    enet_peer_send(server_connection, 0, enet_packet);
}

