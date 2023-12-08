#pragma once
#include "threadworker.h"
#include "image_processing.h"

#define WORK_ENTRIES_MAX 2

class COpenGLDisplay : public CThreadWorker
{
public:
    COpenGLDisplay(const char* name, CameraParams *camera_params, unsigned char *display_buffer); // name is the thread name
    ~COpenGLDisplay ();

	bool PushToDisplay(void* imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id);

	//open gl dimensions:
	CameraParams* camera_params;	
	unsigned char* display_buffer;
	FrameGPU frame_original; // frame on gpu device 
	Debayer debayer;

private: 
	virtual void ThreadRunning(); // overides of COffThreadMachine for worker thread
private:	
	WORKER_ENTRY workerEntries[WORK_ENTRIES_MAX];
	WORKER_ENTRY* workerEntriesFreeQueue[WORK_ENTRIES_MAX];
	int workerEntriesFreeQueueCount;
};