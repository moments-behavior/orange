#ifndef ORANGE_THREADS
#define ORANGE_THREADS
#include <stdint.h>
#include <time.h>
#include <atomic>
#include <memory>
#include <queue> // Required for std::queue
#include "genericmutex.h" 

inline float tick()
{
    struct timespec ts;
    uint32_t res = clock_gettime(CLOCK_MONOTONIC, &ts);
    if (res == -1)
    {
        return 0;
    }
    return ((float)((ts.tv_sec * 1e9) + ts.tv_nsec)) / (float)1.0e9;
}

inline uint64_t sync_fetch_and_add(volatile uint64_t *x, uint64_t by)
{
    return __sync_fetch_and_add(x, by);
}

// A simple, thread-safe queue using the CGenericMutex
template <typename T>
class SafeQueue
{
public:
    void push(T val)
    {
        mutex.Lock();
        queue.push(val);
        mutex.Unlock();
    }

    bool pop(T& val)
    {
        bool success = false;
        mutex.Lock();
        if (!queue.empty())
        {
            val = queue.front();
            queue.pop();
            success = true;
        }
        mutex.Unlock();
        return success;
    }

private:
    std::queue<T> queue;
    CGenericMutex mutex;
};

// The original lock_free_queue (buggy, but kept to avoid breaking other code that might use it)
template <typename T>
class lock_free_queue
{
private:
    struct node
    {
        std::shared_ptr<T> data;
        node *next;
        node() : next(nullptr)
        {
        }
    };
    std::atomic<node *> head;
    std::atomic<node *> tail;
    node *pop_head()
    {
        node *const old_head = head.load();
        if (old_head == tail.load())
        {
            return nullptr;
        }
        head.store(old_head->next);
        return old_head;
    }

public:
    lock_free_queue() : head(new node), tail(head.load())
    {
    }
    lock_free_queue(const lock_free_queue &other) = delete;
    lock_free_queue &operator=(const lock_free_queue &other) = delete;
    ~lock_free_queue()
    {
        while (node *const old_head = head.load())
        {
            head.store(old_head->next);
            delete old_head;
        }
    }
    std::shared_ptr<T> pop()
    {
        node *old_head = pop_head();
        if (!old_head)
        {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> const res(old_head->data);
        delete old_head;
        return res;
    }
    void push(T new_value)
    {
        std::shared_ptr<T> new_data(std::make_shared<T>(new_value));
        node *p = new node;
        node *const old_tail = tail.load();
        old_tail->data.swap(new_data);
        old_tail->next = p;
        tail.store(p);
    }
};

#endif // ORANGE_THREADS