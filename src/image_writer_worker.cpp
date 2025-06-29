// src/image_writer_worker.cpp

#include "image_writer_worker.h"
#include "cuda_context_debug.h"   // Provides CUDA_CONTEXT_SCOPE for robust context handling
#include "NvEncoder/NvCodecUtils.h" // For the 'ck' macro for CUDA error checking
#include <iostream>
#include <opencv2/opencv.hpp>

ImageWriterWorker::ImageWriterWorker(const char* name, CUcontext cuda_context)
    : CThreadWorker(name), m_cuContext(cuda_context) {}

ImageWriterWorker::~ImageWriterWorker() {
    // This destructor logic is a fallback. Ideally, the queue should be empty
    // when the worker is stopped and destroyed.
    while (true) {
        ImageWriter_Entry* entry = GetObjectFromQueueIn();
        if (!entry) {
            break; // Queue is empty
        }

        // Just clean up the event and entry to prevent leaks on exit.
        // We won't try to write the image here as the context might be invalid.
        std::cerr << "Warning: ImageWriterWorker destructor is cleaning up a pending job for: " << entry->file_path << std::endl;
        // Use event_ptr and dereference it
        if (entry->event_ptr) {
            cudaEventDestroy(*entry->event_ptr); // Clean up the CUDA event
            delete entry->event_ptr; // Clean up the pointer itself
        }
        delete entry;
    }
}

bool ImageWriterWorker::WorkerFunction(ImageWriter_Entry* entry) {
    if (!entry) return false;

    // Set the CUDA context for this thread. This is crucial for CUDA calls.
    CUDA_CONTEXT_SCOPE(m_cuContext);

    if (entry->event_ptr) {
        // Block THIS thread until the CUDA memory copy from the acquire_frames thread is complete.
        ck(cudaEventSynchronize(*entry->event_ptr));

        // The event is a one-time use signal, so we destroy it and its memory after synchronization.
        ck(cudaEventDestroy(*entry->event_ptr));
        delete entry->event_ptr;
        entry->event_ptr = nullptr; // Avoid dangling pointer
    } else {
        std::cerr << "Warning: ImageWriterWorker received a job with no CUDA event." << std::endl;
    }


    try {
        // Now that the data is guaranteed to be ready on the CPU, create the cv::Mat.
        cv::Mat image(entry->height, entry->width, CV_8UC3, entry->cpu_buffer);
        // Write the image to disk. This is a blocking I/O call.
        cv::imwrite(entry->file_path, image);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error writing image " << entry->file_path << ": " << ex.what() << std::endl;
    }

    // The entry has been processed, so we are responsible for deleting its container.
    delete entry;

    // This worker is a final destination, so it does not pass data to an output queue.
    return false;
}