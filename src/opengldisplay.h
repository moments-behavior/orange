#pragma once
#include "threadworker.h"
#include "video_capture.h"
#include <nppi.h>

#define WORK_ENTRIES_MAX 10
typedef struct {
	void* imagePtr; // source image buffer
	size_t bufferSize; // size of imagePtr in bytes
	int width;
	int height;
	int pixelFormat;
} WORKER_ENTRY;

struct FrameGPU
{
    unsigned char *d_orig;
    int size_pic;
};

struct Debayer
{
    unsigned char *d_debayer;
    NppiSize size;
    Npp8u nAlpha;
    NppiRect roi;
    NppiBayerGridPosition grid;
};

class COpenGLDisplay : public CThreadWorker
{
public:
    COpenGLDisplay(const char* name, CameraParams *camera_params, unsigned char *display_buffer); // name is the thread name
    ~COpenGLDisplay ();

	bool PushToDisplay(void* imagePtr, size_t bufferSize, int width, int height, int pixelFormat);

	//open gl dimensions:
	CameraParams* camera_params;
	CameraEmergent* ecam;
	
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