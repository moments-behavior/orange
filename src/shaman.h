#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <signal.h>
#include <chrono>
#include <thread>

inline uint64_t get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

namespace shaman {

constexpr const char* SHM_NAME = "/shm_box_vector";
constexpr size_t MAX_OBJECTS = 100;
constexpr size_t MAX_KEYPOINTS = 32;
constexpr size_t QUEUE_SIZE = 8;

struct Data {
    float x, y, z, width, height;
};

struct Object {
    Data data;
    int label;
    float prob;
    float kps[MAX_KEYPOINTS];
    size_t num_kps;
};

struct alignas(64) VectorSlot {
    std::atomic<size_t> count{0};
    std::atomic<uint64_t> timestamp_us{0};
    Object objects[MAX_OBJECTS];
    char padding[64 - ((sizeof(std::atomic<size_t>) + sizeof(std::atomic<uint64_t>) + sizeof(Object) * MAX_OBJECTS) % 64)];
};

struct alignas(64) SharedQueue {
    alignas(64) std::atomic<bool> initialized{false};
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};
    VectorSlot queue[QUEUE_SIZE];
};

class SharedBoxQueue {
    public:
    SharedBoxQueue(bool is_writer) : writer(is_writer) {
        shm_fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error(std::string("shm_open failed: ") + std::strerror(errno));
        }

        struct stat shm_stat;
        if (fstat(shm_fd, &shm_stat) == -1)
            throw std::runtime_error("fstat failed");

        if (shm_stat.st_size < (off_t)sizeof(SharedQueue)) {
            if (ftruncate(shm_fd, sizeof(SharedQueue)) == -1)
                throw std::runtime_error("ftruncate failed");
            creator = true;
        }

        #ifdef __linux__
        shared = static_cast<SharedQueue*>(mmap(nullptr, sizeof(SharedQueue),
                        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_HUGETLB, shm_fd, 0));
        
        if (shared == MAP_FAILED) {
            shared = static_cast<SharedQueue*>(mmap(nullptr, sizeof(SharedQueue),
                            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, shm_fd, 0));
        }
        #else
        shared = static_cast<SharedQueue*>(mmap(nullptr, sizeof(SharedQueue),
                        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
        #endif
        
        if (shared == MAP_FAILED)
            throw std::runtime_error("mmap failed");

        mlock(shared, sizeof(SharedQueue));

        if (creator) {
            new (shared) SharedQueue();
            shared->initialized.store(true, std::memory_order_release);
        } else {
            while (!shared->initialized.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
        
    }
 
    ~SharedBoxQueue() {
        if (shared) {
            munlock(shared, sizeof(SharedQueue));
            munmap(shared, sizeof(SharedQueue));
        }
        if (shm_fd >= 0) close(shm_fd);
    }
    
        bool push(const std::vector<Object>& vec) {
            if (!writer || vec.size() > MAX_OBJECTS) return false;

            const size_t h = shared->head.load(std::memory_order_relaxed);
            const size_t next = (h + 1) % QUEUE_SIZE;
            
            if (next == shared->tail.load(std::memory_order_acquire)) {
                return false; // queue full
            }

            VectorSlot& slot = shared->queue[h];
            
            const size_t count = vec.size();
            std::memcpy(slot.objects, vec.data(), count * sizeof(Object));
            
            slot.timestamp_us.store(get_time_us(), std::memory_order_relaxed);
            slot.count.store(count, std::memory_order_release);

            shared->head.store(next, std::memory_order_release);
            return true;
        }
    
        bool pop(std::vector<Object>& out) {
            if (writer) return false;

            const size_t h = shared->head.load(std::memory_order_acquire);
            const size_t t = shared->tail.load(std::memory_order_relaxed);

            if (h == t) return false; // empty

            const VectorSlot& slot = shared->queue[t];
            
            const size_t count = slot.count.load(std::memory_order_acquire);
            if (count == 0) return false;
            
            out.resize(count);
            std::memcpy(out.data(), slot.objects, count * sizeof(Object));

            shared->tail.store((t + 1) % QUEUE_SIZE, std::memory_order_release);
            return true;
        }

        // pop with timestamp logging - optimized
        bool pop(std::vector<Object>& out, uint64_t& timestamp_us) {
            if (writer) return false;
        
            const size_t h = shared->head.load(std::memory_order_acquire);
            const size_t t = shared->tail.load(std::memory_order_relaxed);
        
            if (h == t) return false; // empty
        
            const VectorSlot& slot = shared->queue[t];
            
            const size_t count = slot.count.load(std::memory_order_acquire);
            if (count == 0) return false;
            
            out.resize(count);
            std::memcpy(out.data(), slot.objects, count * sizeof(Object));
            timestamp_us = slot.timestamp_us.load(std::memory_order_relaxed);
        
            shared->tail.store((t + 1) % QUEUE_SIZE, std::memory_order_release);
            return true;
        }
    
    private:
        int shm_fd = -1;
        bool writer;
        SharedQueue* shared = nullptr;
        bool creator = false;
    
        void map_shared_memory() {
            void* ptr = mmap(nullptr, sizeof(SharedQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
            shared = reinterpret_cast<SharedQueue*>(ptr);
        }
    };
    

}