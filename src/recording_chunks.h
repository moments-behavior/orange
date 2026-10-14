#ifndef ORANGE_RECORDING_CHUNKS_H
#define ORANGE_RECORDING_CHUNKS_H

// Chunked recording: one long recording is split into consecutive chunks
// (chunk_00000/, chunk_00001/, ...) without stopping the cameras. Every camera
// keeps streaming across a chunk boundary; only its encoder's output file rolls
// over. Each chunk folder looks like a normal orange recording folder
// (Cam<serial>.mp4 + Cam<serial>_meta.csv), so existing tools open it as is.
//
// Every camera cuts on the same exposure, even after dropped frames. Each
// frame gets a sync-frame index n counted from the shared PTP acquisition gate:
//
//   n(first frame) = round((ts_first - gate) / period)
//   n(next frame)  = n(previous) + round((ts - ts_previous) / period)
//
// and chunk = n / frames_per_chunk. Because n advances by the PTP time between
// frames, a camera that lost frames still numbers the next exposure the same
// as the cameras that did not. Corresponding frames of the 8 cameras on the
// rig carry timestamps within ~20 ns of each other (<= 4 us worst case), far
// below the half-period rounding margin. Counting whole frames (rather than
// cutting at fixed PTP times) means every chunk holds exactly
// frames_per_chunk frames and the cut never drifts onto a frame time: the
// camera's real period (16 666 666 ns at "60 fps") is not an exact divisor of
// the chunk length, so a time-based cut would slide ~72 us per 30-min chunk
// across the frame grid and hit a frame after ~2.4 days.

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
    bool enabled() const { return frames_per_chunk > 0; }
    // Chunk index of the next frame (call once per frame, in order).
    // gate_ptp_ns: the shared PTP acquisition gate time, or 0 when there was
    // no gate (single camera); the first frame is then sync frame 0.
    int chunk_of(unsigned long long ts, unsigned long long gate_ptp_ns);
    long long sync_frame() const { return frame_index; }
    long long frames_in_chunk() const { return frames_per_chunk; }
    long long first_frame_offset_from_gate() const { return gate_offset_ns; }

  private:
    long long frames_per_chunk = 0;
    double period_ns = 0.0; // tracks the measured single-frame period
    bool started = false;
    unsigned long long prev_ts = 0;
    long long frame_index = 0;
    long long gate_offset_ns = 0;
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
