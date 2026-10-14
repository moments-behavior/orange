// Synthetic test for ChunkClock (src/recording_chunks.cpp): several cameras,
// PTP timestamps with per-camera offset + jitter, random single drops, burst
// gaps, a camera that loses its first frames, multi-day duration. Every frame
// a camera delivers must land in chunk (true frame index / frames_per_chunk).
//
// Build + run:  tests/run_chunk_clock_test.sh
#include "../src/recording_chunks.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

void chown_to_invoking_user(const std::string &, bool) {}

struct Case {
    const char *name;
    int fps;                // configured frame rate
    long long period_ns;    // the camera's real period
    double chunk_seconds;
    double days;
    int cameras;
    double drop_prob;       // per frame, per camera: lose one frame
    double burst_prob;      // per frame, per camera: lose 2..60 frames
    long long jitter_ns;    // +- uniform timestamp jitter
};

static int run(const Case &c) {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<long long> jit(-c.jitter_ns, c.jitter_ns);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_int_distribution<int> burst(2, 60);

    const unsigned long long gate = 11868059796924725ULL;
    const long long first_offset = 360; // first exposure ~360 ns after gate
    const long long n_frames =
        (long long)(c.days * 86400.0 * 1e9 / c.period_ns);
    const long long fpc = std::llround(c.chunk_seconds * c.fps);

    long long wrong = 0, delivered = 0, dropped = 0;
    long long old_rule_disagree = 0; // time-based rule, cameras vs camera 0
    std::vector<int> old_rule_cam0(0);

    // Old (time-based) rule, per camera: floor((ts - ref + P/2) / D).
    const long long D = std::llround(c.chunk_seconds * 1e9);
    const long long P_nom = std::llround(1e9 / c.fps);
    std::vector<std::vector<int>> old_chunk(c.cameras);

    for (int cam = 0; cam < c.cameras; cam++) {
        ChunkClock clk;
        clk.configure(c.chunk_seconds, c.fps);
        long long cam_offset = 2 + 4 * cam; // constant few-ns skew
        long long skip_until = (cam == 3) ? 2 : 0; // cam 3 loses frames 0,1
        bool have_ref = false;
        unsigned long long ref = 0;
        old_chunk[cam].assign(n_frames, -1);
        for (long long j = 0; j < n_frames; j++) {
            if (j < skip_until) {
                dropped++;
                continue;
            }
            if (j > 0 && u(rng) < c.burst_prob) {
                skip_until = j + burst(rng);
                dropped++;
                continue;
            }
            if (j > 0 && u(rng) < c.drop_prob) {
                dropped++;
                continue;
            }
            unsigned long long ts = gate + first_offset + j * c.period_ns +
                                    cam_offset + jit(rng);
            int chunk = clk.chunk_of(ts, gate);
            delivered++;
            if (chunk != (int)(j / fpc)) {
                if (wrong < 5)
                    printf("  cam %d frame %lld: chunk %d, expected %lld\n",
                           cam, j, chunk, j / fpc);
                wrong++;
            }
            if (!have_ref) { // gate-snapped, as the first implementation did
                have_ref = true;
                long long diff = (long long)(ts - gate);
                ref = ts - std::llround((double)diff / P_nom) * P_nom;
            }
            long long d = (long long)(ts - ref) + P_nom / 2;
            old_chunk[cam][j] = (int)(d / D);
        }
    }
    for (int cam = 1; cam < c.cameras; cam++)
        for (long long j = 0; j < n_frames; j++)
            if (old_chunk[cam][j] >= 0 && old_chunk[0][j] >= 0 &&
                old_chunk[cam][j] != old_chunk[0][j])
                old_rule_disagree++;

    printf("%-34s %lld frames/cam, %lld delivered, %lld dropped: %s "
           "(%lld wrong); old time-based rule: %lld cross-camera "
           "disagreements\n",
           c.name, n_frames, delivered, dropped, wrong ? "FAIL" : "ok", wrong,
           old_rule_disagree);
    return wrong ? 1 : 0;
}

int main() {
    int fails = 0;
    // The rig: "60 fps" is really 16 666 666 ns; 8 cameras.
    fails += run({"60fps, 30 min chunks, 4 days", 60, 16666666, 1800, 4.0, 8,
                  1e-4, 2e-6, 4000});
    fails += run({"60fps, 30 s chunks, 3 days", 60, 16666666, 30, 3.0, 8, 1e-4,
                  2e-6, 4000});
    // A camera that runs slightly off its configured rate.
    fails += run({"60 configured / 59.9 real, 30 s", 60, 16694490, 30, 0.5, 4,
                  1e-4, 2e-5, 4000});
    fails += run({"50fps, 30 min chunks, 2 days", 50, 20000000, 1800, 2.0, 8,
                  1e-4, 2e-6, 4000});
    printf(fails ? "FAIL\n" : "OK\n");
    return fails ? 1 : 0;
}
