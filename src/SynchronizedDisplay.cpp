#include "SynchronizedDisplay.h"

SyncDisplay::SyncDisplay(int num_sync_cameras): num_sync_cameras(num_sync_cameras)
{
    for(int i=0; i < num_sync_cameras; i++) {
        CameraEntry* camera_entry = new CameraEntry();
        m_frames.push_back(camera_entry);
        m_frames_ready.push_back(false);
        m_detection_ready.push_back(false);
    }
}

void SyncDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id, int camera_idx)
{
    m_frames[camera_idx]->imagePtr = imagePtr;
    m_frames[camera_idx]->bufferSize = bufferSize;
    m_frames[camera_idx]->width = width;
    m_frames[camera_idx]->height = height;
    m_frames[camera_idx]->pixelFormat = pixelFormat;
    m_frames[camera_idx]->timestamp = timestamp;
    m_frames[camera_idx]->frame_id = frame_id;
    m_frames_ready[camera_idx] = true;
}

void SyncDisplay::WaitForKick(){
	WaitForCondition(m_nodesKicked);
}


void SyncDisplay::SignalDetectionDone(int nodeNum){
	SignalPerNode(m_detection_ready, m_nodesDone, nodeNum);
}

void SyncDisplay::SyncMain()
{

    while(!m_quitting) {
        switch (m_state) {
        case SYNC_WAIT_FOR_FRAME:
            for (int i = 0; i < num_sync_cameras; i++) {
                if (!m_frames_ready[i]) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            m_state = SYNC_SEND_FRAME;
            break;
        case SYNC_SEND_FRAME:
            for (int i = 0; i < num_sync_cameras; i++) {
                m_frames_ready[i] = false;
            }
            SetCondition(m_nodesKicked);
            m_state = SYNC_WAIT_FOR_DETECTION;
            break;
        case SYNC_WAIT_FOR_DETECTION:
            ResetCondition(m_nodesKicked);
            WaitForCondition(m_nodesDone);
            ResetCondition(m_detection_ready, m_nodesDone);
            m_state = SYNC_WAIT_FOR_FRAME;
            break; 
        }
    }
}