#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "opengldisplay.h"
#include <cuda_runtime_api.h>
#include "NvEncoder/NvCodecUtils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


static inline void initalize_gpu_frame(FrameGPU *frame_original, CameraParams *camera_params)
{
    frame_original->size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char);
    ck(cudaMalloc((void **)&frame_original->d_orig, frame_original->size_pic));
}


static inline void initialize_gpu_debayer(Debayer *debayer, CameraParams *camera_params)
{
    int output_channels = 4;
    int size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char) * output_channels;
    cudaMalloc((void **)&debayer->d_debayer, size_pic);
    cudaMemset(debayer->d_debayer, 0xFF, size_pic);
    
    debayer->size.width = camera_params->width;
    debayer->size.height = camera_params->height;
    debayer->nAlpha = 255;
    debayer->roi.x = 0;
    debayer->roi.y = 0;
    debayer->roi.width = camera_params->width;
    debayer->roi.height = camera_params->height;
    if (camera_params->need_reorder) {
        // 100G camera 
        debayer->grid = NPPI_BAYER_GRBG;
    } else if (camera_params->pixel_format.compare("BayerRG8")==0) {
        debayer->grid = NPPI_BAYER_RGGB;
    } else {
        debayer->grid = NPPI_BAYER_GBRG;
    }
}

static inline void debayer_frame_gpu(CameraParams *camera_params, FrameGPU *frame_original, Debayer *debayer)
{
    const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(frame_original->d_orig,
                                                         camera_params->width * sizeof(unsigned char),
                                                         debayer->size,
                                                         debayer->roi,
                                                         debayer->d_debayer,
                                                         camera_params->width * sizeof(uchar4),
                                                         debayer->grid,
                                                         NPPI_INTER_UNDEFINED,
                                                         debayer->nAlpha);
    if (npp_result != 0)
    {
        std::cout << "\nNPP error %d \n"
                  << npp_result << std::endl;
    }
}

static inline void duplicate_channel_gpu(CameraParams *camera_params, FrameGPU *frame_original, Debayer *debayer)
{
    const NppStatus npp_result = nppiDup_8u_C1AC4R(
        frame_original->d_orig,
        camera_params->width * sizeof(unsigned char),
        debayer->d_debayer,
        camera_params->width * sizeof(uchar4),
        debayer->size);

    if (npp_result != 0)
    {
        std::cout << "\nNPP error %d \n"
                  << npp_result << std::endl;
    }
}


COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params, unsigned char *display_buffer)
    : CThreadWorker(name), camera_params(camera_params), display_buffer(display_buffer)
{
    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = WORK_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++)
    {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
}

COpenGLDisplay::~COpenGLDisplay()
{
}

void COpenGLDisplay::ThreadRunning()
{

    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);

    while(IsMachineOn())
    {
        void* f = GetObjectFromQueueIn();
        if(f) {
            WORKER_ENTRY entry = *(WORKER_ENTRY*)f;
            PutObjectToQueueOut(f);
            
            // copy frame from cpu to gpu
            ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, entry.imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));

            if (camera_params->color){
                debayer_frame_gpu(camera_params, &frame_original, &debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_original, &debayer);
            }

            ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));

        }
        usleep(16000); // sleep for 16ms 
    }
}


bool COpenGLDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height, int pixelFormat)
{
    WORKER_ENTRY *entriesOut[WORK_ENTRIES_MAX]; // entris got out from saver thread, their frames should be returned to driver queue.
    int entriesOutCount = WORK_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount)
    { // return the frames to driver, and put entries back to frameSaveEntriesFreeQueue
        printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__, entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++)
        {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount)
    {
        printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__, workerEntriesFreeQueueCount);
        WORKER_ENTRY *entry = workerEntriesFreeQueue[workerEntriesFreeQueueCount - 1];
        workerEntriesFreeQueueCount--;
        entry->imagePtr = imagePtr;
        entry->bufferSize = bufferSize;
        entry->width = width;
        entry->height = height;
        entry->pixelFormat = pixelFormat;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}