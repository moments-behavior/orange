#include "enet_utils.h"
#include "ctrl_generated.h"
#include <iostream>

#include <sys/time.h>

void send_indigo_trigger_message(AppContext *ctx, const std::string &job) {
    using namespace camnet::v1;
    flatbuffers::FlatBufferBuilder builder(256);

    auto jid = builder.CreateString(job);
    auto msg = CreateServer(builder, Kind_KindCommand,
                            camnet::v1::ServerControl_TRIALTRIGGER, jid,
                            /* major */ 1,
                            /* minor */ 1, CommandBody_NONE, 0, 0);
    builder.Finish(msg);

    std::vector<uint8_t> bytes(builder.GetBufferPointer(),
                               builder.GetBufferPointer() + builder.GetSize());

    uint32_t pid = ctx->peers.get_pid_by_name("indigo");
    if (!pid) {
        std::cerr << "[send_indigo_message] Peer 'indigo' not found\n";
        return;
    }

    Outgoing o;
    o.peer_id = pid;
    o.channel = 0;
    o.flags = 0; // best-effort pub/sub
    o.bytes = std::move(bytes);
    ctx->net.send(o);
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // fprintf(stderr, "%ld.%06ld\n", tv.tv_sec, tv.tv_usec);
}

// void send_indigo_trigger_message(
//     AppContext *ctx, flatbuffers::FlatBufferBuilder &flatb_builder) {
//     using namespace camnet::v1;

//     // Build FlatBuffers message
//     auto jid = flatb_builder.CreateString("detection");
//     auto msg = CreateServer(flatb_builder, Kind_KindCommand,
//                             camnet::v1::ServerControl_TRIALTRIGGER, jid,
//                             /* major */ 1,
//                             /* minor */ 1, CommandBody_NONE, 0, 0);
//     flatb_builder.Finish(msg);

//     // Look up peer by name
//     uint32_t pid = ctx->peers.get_pid_by_name("indigo");
//     if (!pid) {
//         // Optional: log or print a warning
//         std::cerr << "[send_indigo_message] Peer 'indigo' not found\n";
//         return;
//     }

//     // Copy FlatBuffer contents into byte vector
//     std::vector<uint8_t> bytes(flatb_builder.GetBufferPointer(),
//                                flatb_builder.GetBufferPointer() +
//                                    flatb_builder.GetSize());

//     // Prepare and send outgoing packet
//     Outgoing o;
//     o.peer_id = pid;
//     o.channel = 0;
//     o.flags = 0;
//     o.bytes = std::move(bytes);

//     ctx->net.send(o);
// }
