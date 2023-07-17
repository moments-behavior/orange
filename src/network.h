#ifndef ORANGE_NETWORK
#define ORANGE_NETWORK
#include "enet.h"
#include "fetch_generated.h"
#include "realtime_tool.h"

void send_serial_data(ENetHost *server, flatbuffers::FlatBufferBuilder& builder, ArucoMarker3d* marker3d, std::map<unsigned int, cv::Point3f>& yolo_obj_3d)
{              
    // build the buffer
    auto position_ramp = FetchGame::Vec3(marker3d->t_vec.x, marker3d->t_vec.y, marker3d->t_vec.z);
    float orientation_ramp = marker3d->angle_x_axis;
    auto ramp_fb = CreateRamp(builder, &position_ramp, orientation_ramp);


    auto position_ball = FetchGame::Vec3(yolo_obj_3d[0].x, yolo_obj_3d[0].y, yolo_obj_3d[0].z); 
    auto ball_fb = CreateBall(builder, &position_ball);

    FetchGame::SceneBuilder scene_builder(builder);
    scene_builder.add_ramp(ramp_fb);
    scene_builder.add_ball(ball_fb);
    auto scene = scene_builder.Finish();
    builder.Finish(scene);
    uint8_t *buf = builder.GetBufferPointer();
    int size = builder.GetSize();

    ENetPacket *packet = enet_packet_create(buf,
                                            size,
                                            ENET_PACKET_FLAG_RELIABLE);
    /* Send the packet to the peer over channel id 0. */
    /* One could also broadcast the packet by         */
    enet_host_broadcast(server, 0, packet);
    // enet_peer_send(peer, 0, packet);

    // Receive some events
    // enet_host_service(client, &event, 0);
}

#endif