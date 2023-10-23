#include "synchronized_queue.h"

SyncQueue::SyncQueue(int num_sync_cameras)
{
    for(int i=0; i < num_sync_cameras; i++) {
        m_queues.push_back(lock_free_queue<FrameEntry*>());
    }
}

void SyncQueue::SyncMain()
{
    while(!m_quitting) {
        // push into detection every 16ms   
        switch (m_state) {
            case SYNC_SLEEP:
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        

    }
}