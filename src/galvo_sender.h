#ifndef ORANGE_GALVO_SENDER
#define ORANGE_GALVO_SENDER

#include <cstdint>
#include <string>

// UDP sender for the GalvoCam target-streaming protocol implemented by the
// Windows motor-control app (see wchen27/GalvoControl, LINUX_SENDER_GUIDE.md
// and PROTOCOL.md). Streams triangulated 3D targets (world mm) so the
// receiver can aim the pan/tilt mirrors.
struct GalvoSenderParams {
    bool enabled = false;
    bool dummy_mode = false;
    std::string target_ip = "10.102.10.90";
    int target_port = 5005;
};

// (Re)opens the UDP socket to target_ip:target_port. Safe to call again to
// re-point the sender at a new address.
bool galvo_sender_start(const std::string &target_ip, int target_port);
void galvo_sender_stop();
bool galvo_sender_running();

// Sends one target packet. valid=false tells the receiver to hold instead of
// chasing a stale/garbage point; x, y, z are ignored by the receiver in that
// case but are still sent as given. No-op if the sender isn't running.
void galvo_sender_send_target(double x, double y, double z, bool valid,
                              uint32_t target_id = 0);

// v2 tracking packet: sends the RAW capture-time position plus an
// EMA-smoothed velocity (estimated across calls using the capture
// timestamps) and the measured capture->send age. The receiver extrapolates
// to its own aim time -- strictly better than a fixed sender-side lead,
// because it also absorbs inference-time variance and receiver-side
// queueing. capture_mono_ns is CLOCK_MONOTONIC at frame handover (0 = "just
// now"). Call from ONE thread (the 3D detection loop) -- the estimator state
// is not per-caller.
void galvo_sender_send_target_tracked(double x, double y, double z, bool valid,
                                      uint64_t capture_mono_ns,
                                      uint32_t target_id = 0);

// Streams a synthetic moving target instead of real triangulated points, for
// exercising the link/receiver without cameras. Reproduces the same test
// pattern the GalvoControl receiver itself draws when idle (see main.cpp's
// "moving target": a 300mm-radius circle in x/y, 1000mm out in z, traced
// once per ~2*pi seconds). Runs on its own thread at ~200 Hz until stopped.
void galvo_sender_start_dummy_target();
void galvo_sender_stop_dummy_target();
bool galvo_sender_dummy_target_running();

#endif
