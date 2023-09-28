#include <iostream>
#include <thread>
#include <filesystem>
#include <iostream>
#include "network_base.h"
#include "thread.h"
#include "types.h"
#include <cstring>

void quit_process(bool error = false, const std::string &reason = "") {
	enet_deinitialize();  //!!!!!!!!!!!!!!!!!NEW!!!!!!!!!!!!!!
	

	//Show console reason before exit
	if (error) {
		std::cout << reason << std::endl;
		system("PAUSE");
		exit(-1);
	}
}


int main(int argc, char *argv[])
{
    //Initialise ENET for networking  //!!!!!!NEW!!!!!!!!
	if (enet_initialize() != 0)
	{
		quit_process(true, "ENET failed to initialize!");
	}


	ENetPeer* server_connection;
    EnetContext client;
    if(enet_initialize(&client, 0, 1)) {
        printf("Network Initialized!\n");
        server_connection = connect_peer(&client, 10, 123, 1, 142, 3333);
        printf("Connecting to server.\n");
    }

    f32 last_time = tick();
    f32 current_time = tick();

    while(true) {
        current_time = tick();
        //Handle All Incoming Packets and Send any enqued packets, does this need to be on another thread?
		service_network(&client, current_time - last_time, [&](const ENetEvent& evnt)
		{
            switch (evnt.type)
                {
                //New connection request or an existing peer accepted our connection request
                case ENET_EVENT_TYPE_CONNECT:
                    {
                        if (evnt.peer == server_connection)
                        {
                            printf("Network: Successfully connected to server!");

                            //Send a 'hello' packet
                            char* text_data = "Hellooo!";
                            ENetPacket* packet = enet_packet_create(text_data, strlen(text_data) + 1, 0);
                            enet_peer_send(server_connection, 0, packet);
                        }	
                    }
                    break;


                //Server has sent us a new packet
                case ENET_EVENT_TYPE_RECEIVE:
                    {
                        printf ("A packet of length %u containing %s was received from %s on channel %u.\n",
                                evnt.packet -> dataLength,
                                evnt.packet -> data,
                                evnt.peer -> data,
                                evnt.channelID);
                        /* Clean up the packet now that we're done using it. */
                        enet_packet_destroy (evnt.packet);
                    }
                    break;

                //Server has disconnected
                case ENET_EVENT_TYPE_DISCONNECT:
                    {
                        printf("Network: Server has disconnected!");
                    }
                    break;
                }

		});
        last_time = current_time;
        sleep(1);
    }

    quit_process();
    return 0;
}