#pragma once
#include "enet_fb_helpers.h"

struct AppContext {
    ENetGuard guard; // must outlive everything that touches ENet
    EnetRuntime net;
    PeerRegistry peers;
    // FBMessageSender sender{&net, /*channel=*/0, ENET_PACKET_FLAG_RELIABLE};

    AppContext() = default;
    AppContext(const AppContext &) = delete;
    AppContext &operator=(const AppContext &) = delete;
};

void send_indigo_trigger_message(AppContext *ctx,
                                 flatbuffers::FlatBufferBuilder &flatb_builder);
