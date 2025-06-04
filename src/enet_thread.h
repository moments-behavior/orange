// src/enet_thread.h
#ifndef ENET_THREAD_H // Add include guards
#define ENET_THREAD_H

#include "yolo_worker.h"
#include <vector>
#include "message_wrapper_generated.h"
#include "fetch_generated.h"
#include "yolo_payload_generated.h"
#include "global.h"
#include <iostream>
#include "network_base.h" // For EnetContext, ConnectedServer, INDIGOSignalBuilder
#include "project.h"


// Forward declarations if full definitions are not needed in this header
// struct EnetContext;
// struct ConnectedServer;
// class YOLOv8Worker;
// struct INDIGOSignalBuilder;

extern std::vector<YOLOv8Worker*> yolo_workers;
extern ENetPeer* external_data_consumer_peer;

// Only DECLARE the function here
void create_enet_thread(EnetContext* server_enet_context,
                        ConnectedServer* my_servers,
                        INDIGOSignalBuilder* indigo_signal_builder,
                        bool* quit_enet);

#endif // ENET_THREAD_H