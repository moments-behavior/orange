#ifndef ORANGE_RECORDING_CHUNKS_H
#define ORANGE_RECORDING_CHUNKS_H

// Chunked recording: one long recording is split into consecutive chunks
// (chunk_00000/, chunk_00001/, ...) without stopping the cameras. Every camera
// keeps streaming across a chunk boundary; only its encoder's output file rolls
// over. Each chunk folder looks like a normal orange recording folder
// (Cam<serial>.mp4 + Cam<serial>_meta.csv), so existing tools open it as is.
//
// Boundaries are placed in PTP time, not by frame count, so every camera cuts
// on the same exposure even if one of them dropped a frame earlier:
//
//   chunk(ts) = floor((ts - ref + period/2) / chunk_ns)
//
// ref is the shared PTP gate time snapped to the frame phase (the first frame's
// timestamp, measured from the gate, rounded to a whole number of periods).
// Corresponding frames of different cameras carry PTP timestamps within a few
// microseconds of each other, and the boundary sits half a frame period before
// the first frame of the chunk, so all cameras assign each exposure to the same
// chunk.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

std::string chunk_folder_name(const std::string &session_folder, int chunk);

class ChunkClock {
  public:
    // chunk_seconds <= 0 disables chunking (every frame is chunk 0).
    void configure(double chunk_seconds, int frame_rate);
    bool enabled() const { return chunk_ns > 0; }
    // gate_ptp_ns: the shared PTP acquisition gate time, or 0 when there was no
    // gate (single camera); the first frame's timestamp is then the reference.
    void set_reference(unsigned long long first_ts,
                       unsigned long long gate_ptp_ns);
    bool has_reference() const { return reference_set; }
    // Chunk index of a frame. Never decreases.
    int chunk_of(unsigned long long ts);
    unsigned long long reference() const { return ref_ns; }
    long long first_frame_offset_from_gate() const { return gate_offset_ns; }

  private:
    long long chunk_ns = 0;
    long long period_ns = 0;
    unsigned long long ref_ns = 0;
    long long gate_offset_ns = 0;
    bool reference_set = false;
    int last_chunk = 0;
};

// What one camera wrote into one chunk; reported when its files are closed.
struct CameraChunkSummary {
    std::string camera_serial;
    unsigned long long frames = 0;          // packets muxed into the mp4
    unsigned long long metadata_rows = 0;   // rows in _meta.csv
    unsigned long long first_frame_id = 0;  // orange frame_id (session-wide)
    unsigned long long last_frame_id = 0;
    unsigned long long first_timestamp = 0; // camera PTP timestamp (ns)
    unsigned long long last_timestamp = 0;
    bool first_packet_keyframe = false;
    double finalize_ms = 0.0;
};

// Collects per-camera summaries for each chunk of a recording session and,
// once every recording camera has closed a chunk, writes
// <chunk folder>/chunk_complete.json. A downstream copier can treat a chunk
// folder as finished (all files closed, safe to move) when that file exists.
namespace chunk_registry {
void register_camera(const std::string &session_folder);
void camera_finished_chunk(const std::string &session_folder, int chunk,
                           double chunk_seconds,
                           const CameraChunkSummary &summary);
} // namespace chunk_registry

// A single background thread that runs queued jobs in order. Used by each
// encoder to close finished chunk files (mp4 trailer + avio_close + csv close)
// off the encode path. The destructor drains the queue before returning.
class ChunkFinalizer {
  public:
    ChunkFinalizer();
    ~ChunkFinalizer();
    void post(std::function<void()> job);
    size_t pending();

  private:
    void run();
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    bool busy = false;
    bool stopping = false;
    std::thread worker;
};

#endif
