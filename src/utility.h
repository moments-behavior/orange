#ifndef UTILITY
#define UTILITY
// C++ implementation of the above approach
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
  
// Thread-safe queue
template <typename T>
class TSQueue {
private:
    // Underlying queue
    std::queue<T> m_queue;
  
    // mutex for thread synchronization
    std::mutex m_mutex;
  
    // Condition variable for signaling
    std::condition_variable m_cond;
  
public:
    // Pushes an element to the queue
    void push(T item)
    {
  
        // Acquire lock
        std::unique_lock<std::mutex> lock(m_mutex);
  
        // Add item
        m_queue.push(item);
  
        // Notify one thread that
        // is waiting
        m_cond.notify_one();
    }
  
    // Pops an element off the queue
    T pop()
    {
  
        // acquire lock
        std::unique_lock<std::mutex> lock(m_mutex);
  
        // wait until queue is not empty
        m_cond.wait(lock,
                    [this]() { return !m_queue.empty(); });
  
        // retrieve item
        T item = m_queue.front();
        m_queue.pop();
  
        // return item
        return item;
    }
    T size()
    {
        return m_queue.size();
    }
    T empty()
    {
        return m_queue.empty();
    }
    T front()
    {
        return m_queue.front();
    }
};
#endif
