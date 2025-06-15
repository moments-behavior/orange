#ifndef IMAGE_WRITER_WORKER_H
#define IMAGE_WRITER_WORKER_H

#include "threadworker.h"
#include <string>
#include <opencv2/opencv.hpp>

struct ImageWriter_Entry {
    cv::Mat image;
    std::string file_path;
};

// FIX: Templated on the struct 'ImageWriter_Entry' itself, not a pointer to it.
class ImageWriterWorker : public CThreadWorker<ImageWriter_Entry>
{
public:
    ImageWriterWorker(const char* name);
    ~ImageWriterWorker() override;

protected:
    // FIX: The signature now correctly overrides the base class virtual function.
    bool WorkerFunction(ImageWriter_Entry* f) override;
};

#endif // IMAGE_WRITER_WORKER_H