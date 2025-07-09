// src/image_writer_worker.cpp

#include "image_writer_worker.h"
#include "NvEncoder/NvCodecUtils.h" // For the 'ck' macro for CUDA error checking
#include <iostream>
#include <opencv2/opencv.hpp>

ImageWriterWorker::ImageWriterWorker(
    const char* name)
    : CThreadWorker(name) {}

ImageWriterWorker::~ImageWriterWorker() {
    // This logic is mostly fine, but we can simplify the event handling
    // since this worker no longer owns the CUDA event lifecycle.
    while (true) {
        ImageWriter_Entry* entry = GetObjectFromQueueIn();
        if (!entry) {
            break; // Queue is empty
        }
        std::cerr << "Warning: ImageWriterWorker destructor is cleaning up a pending job for: " << entry->file_path << std::endl;

        // The worker doesn't manage events, but we should still clean up the entry
        delete entry;
    }
}

bool ImageWriterWorker::WorkerFunction(ImageWriter_Entry* entry) {
    if (!entry) return false;

    // This thread's only job is to write the provided CPU buffer to disk.
    // It assumes the buffer is already filled and ready for access.

    if (!entry->cpu_buffer) {
        std::cerr << "Error: ImageWriterWorker received a job with a null CPU buffer for: " << entry->file_path << std::endl;
        delete entry;
        return false;
    }

    try {
        // Now that the data is guaranteed to be ready on the CPU, create the cv::Mat.
        // Assuming BGR data for now as per your original code.
        cv::Mat image(entry->height, entry->width, CV_8UC3, entry->cpu_buffer);

        // Write the image to disk. This is a blocking I/O call.
        cv::imwrite(entry->file_path, image);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error writing image " << entry->file_path << ": " << ex.what() << std::endl;
    }

    // The cpu_buffer inside the entry must be freed by whoever allocated it,
    // which should happen after this worker is completely done. For now, we just delete the entry container.
    // A more robust system would use smart pointers or a memory pool.
    delete entry->cpu_buffer; // Assuming the buffer was allocated with new[]
    delete entry;

    return false;
}