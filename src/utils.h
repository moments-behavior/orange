#ifndef ORANGE_UTILS
#define ORANGE_UTILS
#include <atomic>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <npp.h>
#include <stdint.h>
#include <time.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        const cudaError_t error_code = call;                                   \
        if (error_code != cudaSuccess) {                                       \
            printf("CUDA Error:\n");                                           \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error code: %d\n", error_code);                        \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

inline float tick() {
    struct timespec ts;
    uint32_t res = clock_gettime(CLOCK_MONOTONIC, &ts);
    if (res == -1) {
        return 0;
    }
    return ((float)((ts.tv_sec * 1e9) + ts.tv_nsec)) / (float)1.0e9;
}

// Increment value with a lock and return the previous value
inline uint64_t sync_fetch_and_add(volatile uint64_t *x, uint64_t by) {
    // NOTE(dd): we're using a gcc/clang compiler extension to do this
    // because mutexes were for some reason slower
    return __sync_fetch_and_add(x, by);
}

inline NppStreamContext make_npp_stream_context(int device_id,
                                                cudaStream_t stream) {
    NppStreamContext ctx = {};
    cudaDeviceProp prop;

    // Set device
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&prop, device_id);

    ctx.hStream = stream;
    ctx.nCudaDeviceId = device_id;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;

    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                           cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                           cudaDevAttrComputeCapabilityMinor, device_id);

    unsigned int stream_flags = 0;
    cudaStreamGetFlags(stream, &stream_flags);
    ctx.nStreamFlags = stream_flags;

    // DO NOT touch ctx.nReserved0

    return ctx;
}

template <typename T> class lock_free_queue {
  private:
    struct node {
        std::shared_ptr<T> data;
        node *next;
        node() : next(nullptr) {}
    };
    std::atomic<node *> head;
    std::atomic<node *> tail;
    node *pop_head() {
        node *const old_head = head.load();
        if (old_head == tail.load()) {
            return nullptr;
        }
        head.store(old_head->next);
        return old_head;
    }

  public:
    lock_free_queue() : head(new node), tail(head.load()) {}
    lock_free_queue(const lock_free_queue &other) = delete;
    lock_free_queue &operator=(const lock_free_queue &other) = delete;
    ~lock_free_queue() {
        while (node *const old_head = head.load()) {
            head.store(old_head->next);
            delete old_head;
        }
    }
    std::shared_ptr<T> pop() {
        node *old_head = pop_head();
        if (!old_head) {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> const res(old_head->data);
        delete old_head;
        return res;
    }
    void push(T new_value) {
        std::shared_ptr<T> new_data(std::make_shared<T>(new_value));
        node *p = new node;
        node *const old_tail = tail.load();
        old_tail->data.swap(new_data);
        old_tail->next = p;
        tail.store(p);
    }
};

class FPSEstimator {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_time;
    float accumulated_time = 0.0f;
    int frame_count = 0;
    float report_interval = 0.5f; // seconds
    float last_fps = 0.0f;

  public:
    FPSEstimator() { start_time = Clock::now(); }

    // Call this once per frame
    void update() {
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - start_time).count();
        start_time = now;

        accumulated_time += dt;
        frame_count++;

        if (accumulated_time >= report_interval) {
            last_fps = frame_count / accumulated_time;
            accumulated_time = 0.0f;
            frame_count = 0;
        }
    }

    float get_fps() const { return last_fps; }

    void reset() {
        start_time = Clock::now();
        accumulated_time = 0.0f;
        frame_count = 0;
        last_fps = 0.0f;
    }
};

#endif // ORANGE_THREADS
