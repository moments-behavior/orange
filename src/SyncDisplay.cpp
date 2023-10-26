#include "SyncDisplay.h"

SyncDisplay::SyncDisplay(int num_sync_cameras): num_sync_cameras(num_sync_cameras)
{
    for(int i=0; i < num_sync_cameras; i++) {
        CameraEntry* camera_entry = new CameraEntry();
        m_frames.push_back(camera_entry);
        m_frames_ready.push_back(false);
        m_detection_ready.push_back(false);
        m_axisSentMove.push_back(false);
    }
    m_nodesKicked = false;
    m_nodesDone = false;
    m_nodesMoved = false;
    m_triangulation_started = false;
    m_triangulation_in_proc = false;
    m_triangulation_done = false;
    m_state = F_SYNC_WAIT_FOR_FRAME;
    m_quitting = false;
}

void SyncDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id, int camera_idx)
{
    // check the state 
    if (m_state == F_SYNC_WAIT_FOR_FRAME) {
        m_frames[camera_idx]->imagePtr = imagePtr;
        m_frames[camera_idx]->bufferSize = bufferSize;
        m_frames[camera_idx]->width = width;
        m_frames[camera_idx]->height = height;
        m_frames[camera_idx]->pixelFormat = pixelFormat;
        m_frames[camera_idx]->timestamp = timestamp;
        m_frames[camera_idx]->frame_id = frame_id;
        m_frames_ready[camera_idx] = true;
    }
}

void SyncDisplay::WaitForKick(){
    WaitForCondition(m_nodesKicked);
}

void SyncDisplay::SignalMoveSent(int nodeNum){
	SignalPerNode(m_axisSentMove, m_nodesMoved, nodeNum);
}

void SyncDisplay::SignalTriangulationDone() {
    SetCondition(m_triangulation_done);
}


void SyncDisplay::SignalTriangulationInProc() {
    SetCondition(m_triangulation_in_proc);
}

void SyncDisplay::SignalDetectionDone(int nodeNum){
    SignalPerNode(m_detection_ready, m_nodesDone, nodeNum);
}

void SyncDisplay::WaitForTriangulation() {
    WaitForCondition(m_triangulation_started);
}

void SyncDisplay::SyncMain()
{

    while(!m_quitting) {
        printf(str_sync_states[m_state]);
        printf("\n");

        switch (m_state) {
        case F_SYNC_WAIT_FOR_FRAME:
            for (int i = 0; i < num_sync_cameras; i++) {
                while (!m_frames_ready[i]) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            m_state = F_SYNC_SEND_FRAME;
            break;
        case F_SYNC_SEND_FRAME:
            for (int i = 0; i < num_sync_cameras; i++) {
                m_frames_ready[i] = false;
            }
            SetCondition(m_nodesKicked);
            m_state = F_SYNC_DETECTION_STARTED;
            break;
        case F_SYNC_DETECTION_STARTED:
            // wait for condition of saying nodes kicked
            WaitForCondition(m_nodesMoved);
            ResetCondition(m_nodesKicked);
            ResetCondition(m_axisSentMove, m_nodesMoved);
            m_state = F_SYNC_WAIT_FOR_DETECTION;
            break;
        case F_SYNC_WAIT_FOR_DETECTION:
            WaitForCondition(m_nodesDone);
            ResetCondition(m_detection_ready, m_nodesDone);
            m_state = F_SYNC_START_TRIANGULATION;
            break; 
        case F_SYNC_START_TRIANGULATION:
            SetCondition(m_triangulation_started);
            m_state = F_SYNC_TRIANGULATION_IN_PROC;
            break;
        case F_SYNC_TRIANGULATION_IN_PROC:
            WaitForCondition(m_triangulation_in_proc);
            ResetCondition(m_triangulation_started);
            ResetCondition(m_triangulation_in_proc);
            m_state = F_SYNC_WAIT_FOR_TRIANGULATION;
            break;
        case F_SYNC_WAIT_FOR_TRIANGULATION:
            WaitForCondition(m_triangulation_done);
            ResetCondition(m_triangulation_done);
            m_state = F_SYNC_WAIT_FOR_FRAME;
            break;
        }
    }
}