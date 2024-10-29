#ifndef ORANGE_GPU_VIDEO_ENCODER
#define ORANGE_GPU_VIDEO_ENCODER

#include "threadworker.h"
#include "video_capture.h"
#include "FFmpegWriter.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderCLIOptions.h"
#include "image_processing.h"

#define ENCODER_ENTRIES_MAX 20

struct Writer
{
    std::string video_file;
    std::string keyframe_file;
	std::string metadata_file;
    FFmpegWriter *video;
    std::ofstream* metadata;
};

struct EncoderContext
{
    NV_ENC_BUFFER_FORMAT eFormat;
    NvEncoderInitParam encodeCLIOptions;
    CUcontext cuContext;
    unsigned long long num_frame_encode = 0;
    std::vector<std::vector<uint8_t>> vPacket;
    NvEncoderCuda *pEnc;
};

class GPUVideoEncoder : public CThreadWorker
{
public:
    GPUVideoEncoder(const char* name, CameraParams *camera_params, std::string encoder_setup, std::string folder_name, bool* encoder_ready_signal); // name is the thread name
    ~GPUVideoEncoder ();

	bool PushToDisplay(void* imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id, uint64_t timestamp_sys);
	void ProcessOneFrame(void *f);

	//open gl dimensions:
	bool* encoder_ready_signal;
	CameraParams* camera_params;	
	unsigned char* display_buffer;
	FrameGPU frame_original; // frame on gpu device 
	Debayer debayer;

	// encoding
	EncoderContext encoder;
    Writer writer;
	std::string encoder_setup;
	std::string folder_name;

private: 
	virtual void ThreadRunning(); // overides of COffThreadMachine for worker thread
private:	
	WORKER_ENTRY workerEntries[ENCODER_ENTRIES_MAX];
	WORKER_ENTRY* workerEntriesFreeQueue[ENCODER_ENTRIES_MAX];
	int workerEntriesFreeQueueCount;
};

#endif