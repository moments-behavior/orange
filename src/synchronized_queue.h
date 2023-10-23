#ifndef SynchronizedQueue
#define SynchronizedQueue

#include <vector.h>
#include "thread.h"
#include <chrono>
#include <thread>

struct FrameEntry{
    void* imagePtr; // source image buffer
    size_t bufferSize; // size of imagePtr in bytes
    int width;
    int height;
    int pixelFormat;
    unsigned long long timestamp;
    unsigned long long frame_id;
};

class SyncQueue
{
public:
    SyncQueue(int num_sync_cameras);
    void CreateThread(){
        m_thread = std::thread(&SyncQueue::SyncMain, this);
    }

private:
    std::vector<lock_free_queue<FrameEntry*>> m_queues;
    std::thread m_thread;

	enum SyncStateEnum {
		SYNC_WAIT,
		SYNC_PUSH_FRAME,
		SYNC_EXITED
	};

    SyncStateEnum m_state;
    bool m_quitting;
	void SyncMain(); // the work for this thread
};

#endif