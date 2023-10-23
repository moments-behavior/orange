#ifndef SynchronizedDisplay
#define SynchronizedDisplay

#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

struct CameraEntry{
    void* imagePtr; // source image buffer
    size_t bufferSize; // size of imagePtr in bytes
    int width;
    int height;
    int pixelFormat;
    unsigned long long timestamp;
    unsigned long long frame_id;
};

class SyncDisplay
{
public:
    SyncDisplay(int num_sync_cameras);
    void CreateThread(){
        m_thread = std::thread(&SyncDisplay::SyncMain, this);
    }
    void PushToDisplay(void* imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id, int camera_idx);
	void WaitForKick();	
    void SignalDetectionDone(int nodeNum);

private:
	std::mutex m_mutex;
	std::condition_variable m_cond;

    bool m_nodesKicked, m_nodesDone;

	// synchronization variables
    std::thread m_thread;
    std::vector<CameraEntry*> m_frames;
    std::vector<bool> m_frames_ready;
    std::vector<bool> m_detection_ready;


	enum SyncStateEnum {
		SYNC_WAIT_FOR_FRAME,
        SYNC_SEND_FRAME,
        SYNC_WAIT_FOR_DETECTION
    };

    SyncStateEnum m_state;
    bool m_quitting;
    int num_sync_cameras;
	void SyncMain(); // the work for this thread

    // Mark that a particular node has signalled back for the given condition
	void SignalPerNode(std::vector<bool> &nodeFlags, 
					   bool &condition, 
					   int nodeNum){
		std::unique_lock<std::mutex> lock(m_mutex);
		nodeFlags.at(nodeNum) = true;
		bool gotAll = true;
		for (int iAxis = 0; iAxis < nodeFlags.size(); iAxis++)
			gotAll = gotAll && nodeFlags.at(iAxis);
		if (gotAll){
			condition = true;
			m_cond.notify_all();
		}
	}

	// Set a condition that doesn't rely on all the nodes to do something
	void SetCondition(bool &condition){
		std::unique_lock<std::mutex> lock(m_mutex);
		condition = true;
		m_cond.notify_all();
	}

	// Wait for a given condition
	void WaitForCondition(bool &condition){
		std::unique_lock<std::mutex> lock(m_mutex);
		while (!condition && !m_quitting)
			m_cond.wait(lock);
	}

	// Clear the indicators for a particular condition
	void ResetCondition(std::vector<bool> &nodeFlags, bool &condition){
		std::unique_lock<std::mutex> lock(m_mutex);
		for (int iAxis = 0; iAxis < nodeFlags.size(); iAxis++)
			nodeFlags.at(iAxis) = false;
		condition = false;
	}

	// Clear a condition that doesn't relay on all the nodes
	void ResetCondition(bool &condition){
		std::unique_lock<std::mutex> lock(m_mutex);
		condition = false;
	}

};

#endif