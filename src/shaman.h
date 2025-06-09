#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <vector>
#include <iostream>

namespace shaman {

constexpr const char* SHM_NAME = "/shm_box_vector";
constexpr size_t MAX_OBJECTS = 100;
constexpr size_t MAX_KEYPOINTS = 32;
constexpr size_t QUEUE_SIZE = 64;

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
            if (writer) {
                // Wait for reader
                while ((shm_fd = shm_open(SHM_NAME, O_RDWR, 0666)) == -1) {
                    usleep(100000);
                }
                map_shared_memory();
                while (!shared->initialized.load(std::memory_order_acquire)) {
                    usleep(10000);
                }
            } else {
                shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
                if (shm_fd == -1) throw std::runtime_error("shm_open failed");
    
                ftruncate(shm_fd, sizeof(SharedQueue));
                map_shared_memory();
    
                std::memset(shared, 0, sizeof(SharedQueue));
                shared->head.store(0, std::memory_order_relaxed);
                shared->tail.store(0, std::memory_order_relaxed);
                shared->initialized.store(true, std::memory_order_release);
            }
        }
    
        ~SharedBoxQueue() {
            if (shared) munmap(shared, sizeof(SharedQueue));
            if (shm_fd >= 0) {
                close(shm_fd);
                if (!writer) shm_unlink(SHM_NAME);
            }
        }
    
        // Writer pushes a vector
        bool push(const std::vector<Object>& vec) {
            if (!writer) return false;
            if (vec.size() > MAX_OBJECTS) return false;
    
            size_t h = shared->head.load(std::memory_order_relaxed);
            size_t t = shared->tail.load(std::memory_order_acquire);
            size_t next = (h + 1) % QUEUE_SIZE;
    
            if (next == t) return false; // queue full
    
            // Copy data into shared memory
            VectorSlot& slot = shared->queue[h];
            slot.count = vec.size();
            for (size_t i = 0; i < vec.size(); ++i) {
                slot.objects[i] = vec[i];
            }
    
            shared->head.store(next, std::memory_order_release);
            return true;
        }
    
        // Reader pops a vector
        bool pop(std::vector<Object>& out) {
            if (writer) return false;
    
            size_t h = shared->head.load(std::memory_order_acquire);
            size_t t = shared->tail.load(std::memory_order_relaxed);
    
            if (h == t) return false; // empty
    
            const VectorSlot& slot = shared->queue[t];
            out.resize(slot.count);
            for (size_t i = 0; i < slot.count; ++i) {
                out[i] = slot.objects[i];
            }
    
            shared->tail.store((t + 1) % QUEUE_SIZE, std::memory_order_release);
            return true;
        }
    
    private:
        int shm_fd = -1;
        bool writer;
        SharedQueue* shared = nullptr;
    
        void map_shared_memory() {
            void* ptr = mmap(nullptr, sizeof(SharedQueue), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
            shared = reinterpret_cast<SharedQueue*>(ptr);
        }
    };
    

}