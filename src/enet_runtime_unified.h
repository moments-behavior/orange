// enet_runtime_unified.h
#pragma once
#include <thread>
#include <chrono>
#include "enet_runtime_select.h"  // typedef EnetRuntime to threaded or inline
#include "enet_types.h"

// Always-return-bool connect() for both runtimes.
// - Threaded: enqueue connect request and return true.
// - Inline:   perform connect and return result.
inline bool enet_connect(EnetRuntime& rt, const ConnectReq& cr) {
#if defined(ENET_RUNTIME_THREADED)
    rt.connect(cr);
    return true; // enqueued
#else
    return rt.connect(cr);
#endif
}

// Drain all pending events *now*, calling handler(evt) for each.
// Returns number of events dispatched (0 if none).
inline int enet_dispatch_drain(EnetRuntime& rt, void (*handler)(const Incoming&)) {
#if defined(ENET_RUNTIME_THREADED)
    int n = 0;
    Incoming evt;
    while (rt.poll(evt)) {   // non-blocking dequeue
        handler(evt);
        ++n;
    }
    return n;
#else
    // Inline executes the handler inside step().
    // Non-blocking behavior: timeout_ms = 0.
    rt.step(0, handler);
    return 0; // step() doesn't report count
#endif
}

// Block up to timeout_ms waiting for *first* event, then drain.
// Handy if you want a blocking dispatcher loop that works for both runtimes.
inline int enet_dispatch_block(EnetRuntime& rt, int timeout_ms,
                               void (*handler)(const Incoming&)) {
#if defined(ENET_RUNTIME_THREADED)
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    int total = 0;
    for (;;) {
        total += enet_dispatch_drain(rt, handler);
        if (total > 0) break;
        if (timeout_ms <= 0) break;
        if (steady_clock::now() >= deadline) break;
        std::this_thread::sleep_for(milliseconds(1));
    }
    return total;
#else
    rt.step(timeout_ms, handler);  // inline does the waiting internally
    return 0;
#endif
}

// Ensure client-mode I/O thread exists for threaded runtime; no-op for inline.
inline bool enet_start_client(EnetRuntime& rt,
                              size_t channels = 2,
                              enet_uint32 in_bw = 0,
                              enet_uint32 out_bw = 0) {
#if defined(ENET_RUNTIME_THREADED)
    return rt.start_client(channels, in_bw, out_bw);
#else
    (void)rt; (void)channels; (void)in_bw; (void)out_bw;
    return true;
#endif
}
