#pragma once
#include "enet_runtime_threaded.h"
#include "enet_types.h"
#include <chrono>
#include <thread>

// - Threaded: enqueue connect request and return true.
inline bool enet_connect(EnetRuntime &rt, const ConnectReq &cr) {
    rt.connect(cr);
    return true; // enqueued
}

// Drain all pending events *now*, calling handler(evt) for each.
// Returns number of events dispatched (0 if none).
inline int enet_dispatch_drain(EnetRuntime &rt,
                               void (*handler)(const Incoming &)) {
    int n = 0;
    Incoming evt;
    while (rt.poll(evt)) { // non-blocking dequeue
        handler(evt);
        ++n;
    }
    return n;
}

// Block up to timeout_ms waiting for *first* event, then drain.
inline int enet_dispatch_block(EnetRuntime &rt, int timeout_ms,
                               void (*handler)(const Incoming &)) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    int total = 0;
    for (;;) {
        total += enet_dispatch_drain(rt, handler);
        if (total > 0)
            break;
        if (timeout_ms <= 0)
            break;
        if (steady_clock::now() >= deadline)
            break;
        std::this_thread::sleep_for(milliseconds(1));
    }
    return total;
}

// Ensure client-mode I/O thread exists for threaded runtime;
inline bool enet_start_client(EnetRuntime &rt, size_t channels = 2,
                              enet_uint32 in_bw = 0, enet_uint32 out_bw = 0) {
    return rt.start_client(channels, in_bw, out_bw);
}
