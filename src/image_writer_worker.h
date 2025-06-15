// src/image_writer_worker.h

#ifndef IMAGE_WRITER_WORKER_H
#define IMAGE_WRITER_WORKER_H

#include "threadworker.h"
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h> // For cudaEvent_t
#include <cuda.h>         // For CUcontext

// This is now the single, authoritative definition for ImageWriter_Entry
struct ImageWriter_Entry {
    std::string file_path;
    cudaEvent_t event;
    unsigned char* cpu_buffer;
    int width;
    int height;
};

class ImageWriterWorker : public CThreadWorker<ImageWriter_Entry>
{
public:
    ImageWriterWorker(const char* name, CUcontext cuda_context);
    ~ImageWriterWorker() override;

protected:
    bool WorkerFunction(ImageWriter_Entry* f) override;

private:
    CUcontext m_cuContext;
};

#endif // IMAGE_WRITER_WORKER_H