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

struct Rect {
    float x, y, width, height;
};

struct Object {
    Rect rect;
    int label;
    float prob;
    float kps[MAX_KEYPOINTS];
    size_t num_kps;
};

struct VectorSlot {
    size_t count;
    Object objects[MAX_OBJECTS];
    uint64_t timestamp_us;
    uint64_t frame_id;
    uint16_t camera_id;
};

struct SharedQueue {
    std::atomic<bool> initialized;
    std::atomic<size_t> head;  // writer
    std::atomic<size_t> tail;  // reader
    VectorSlot queue[QUEUE_SIZE];
};

class SharedBoxQueue {
    public:
    SharedBoxQueue(bool is_writer) : writer(is_writer) {
        // Try to create shared memory first
        shm_fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error(std::string("shm_open failed: ") + std::strerror(errno));
        }

        // Check current size
        struct stat shm_stat;
        if (fstat(shm_fd, &shm_stat) == -1)
            throw std::runtime_error("fstat failed");

        if (shm_stat.st_size < (off_t)sizeof(SharedQueue)) {
            // First process to run
            if (ftruncate(shm_fd, sizeof(SharedQueue)) == -1)
                throw std::runtime_error("ftruncate failed");
            creator = true;
        }

        // Map the memory AFTER truncating
        shared = static_cast<SharedQueue*>(mmap(nullptr, sizeof(SharedQueue),
                        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
        if (shared == MAP_FAILED)
            throw std::runtime_error("mmap failed");

        if (creator) {
            std::memset(shared, 0, sizeof(SharedQueue));
            shared->head.store(0, std::memory_order_relaxed);
            shared->tail.store(0, std::memory_order_relaxed);
            shared->initialized.store(true, std::memory_order_release);
        } else {
            // Wait for initialization
            size_t attempts = 0;
            while (!shared->initialized.load(std::memory_order_acquire)) {
                if (++attempts > 1000)
                    throw std::runtime_error("Timeout waiting for shared memory init");
                usleep(1000); // wait 1ms
            }
        }
        
    }
 
    ~SharedBoxQueue() {
        if (shared) munmap(shared, sizeof(SharedQueue));
        if (shm_fd >= 0) close(shm_fd);
        // if (!writer) shm_unlink(SHM_NAME);
    }
    
        // Writer pushes a vector
        bool push(const std::vector<Object>& vec, uint64_t frame_id, uint16_t camera_id) {
            if (!writer) return false;
            if (vec.size() > MAX_OBJECTS) return false;

            std::cout << "Camera ID: " << camera_id << ", Frame ID: " << frame_id << std::endl;
            if (!vec.empty()) {
                std::cout << "Detected objects:" << std::endl;
                for (const auto& obj : vec) {
                    std::cout << "  Label: " << obj.label 
                              << ", Probability: " << obj.prob 
                              << ", Rect: [" << obj.rect.x << ", " << obj.rect.y 
                              << ", " << obj.rect.width << ", " << obj.rect.height << "]" 
                              << std::endl;
                }
            }

            size_t h = shared->head.load(std::memory_order_relaxed);
            size_t t = shared->tail.load(std::memory_order_acquire);
            size_t next = (h + 1) % QUEUE_SIZE;

            if (next == t) return false; // queue full

            VectorSlot& slot = shared->queue[h];
            slot.count = vec.size();
            for (size_t i = 0; i < vec.size(); ++i) {
                slot.objects[i] = vec[i];
            }

            slot.timestamp_us = get_time_us(); 
            slot.frame_id = frame_id;
            slot.camera_id = camera_id;

            shared->head.store(next, std::memory_order_release);
            return true;
        }
    
        // Reader pops a vector
        bool pop(std::vector<Object>& out, uint64_t& frame_id) {
            if (writer) return false;
    
            size_t h = shared->head.load(std::memory_order_acquire);
            size_t t = shared->tail.load(std::memory_order_relaxed);
    
            if (h == t) return false; // empty
    
            const VectorSlot& slot = shared->queue[t];
            out.resize(slot.count);
            for (size_t i = 0; i < slot.count; ++i) {
                out[i] = slot.objects[i];
            }
            frame_id = slot.frame_id;
    
            shared->tail.store((t + 1) % QUEUE_SIZE, std::memory_order_release);
            return true;
        }

        // pop with timestamp logging
        bool pop(std::vector<Object>& out, uint64_t& timestamp_us, uint64_t& frame_id, uint16_t& camera_id) {
            if (writer) return false;
        
            size_t h = shared->head.load(std::memory_order_acquire);
            size_t t = shared->tail.load(std::memory_order_relaxed);
        
            if (h == t) return false; // empty
        
            const VectorSlot& slot = shared->queue[t];
            out.resize(slot.count);
            for (size_t i = 0; i < slot.count; ++i) {
                out[i] = slot.objects[i];
            }
        
            timestamp_us = slot.timestamp_us;  
            frame_id = slot.frame_id;
            camera_id = slot.camera_id;
        
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