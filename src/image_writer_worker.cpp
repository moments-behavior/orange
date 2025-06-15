#include "image_writer_worker.h"
#include <iostream>

ImageWriterWorker::ImageWriterWorker(const char* name) : CThreadWorker(name) {}

ImageWriterWorker::~ImageWriterWorker() {
    // Ensure all pending images are written before shutting down
    while (true) {
        ImageWriter_Entry* entry = GetObjectFromQueueIn();
        if (!entry) {
            break; // Queue is empty
        }
        
        try {
            cv::imwrite(entry->file_path, entry->image);
            std::cout << "Saved pending image on exit: " << entry->file_path << std::endl;
        } catch (const cv::Exception& ex) {
            std::cerr << "Error writing pending image " << entry->file_path << ": " << ex.what() << std::endl;
        }
        delete entry;
    }
}

// FIX: The signature now correctly matches the header and the base class.
bool ImageWriterWorker::WorkerFunction(ImageWriter_Entry* entry) {
    if (!entry) return false;

    try {
        cv::imwrite(entry->file_path, entry->image);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error writing image " << entry->file_path << ": " << ex.what() << std::endl;
    }

    // The entry has been processed, so we are responsible for deleting the memory.
    delete entry;
    
    // Return false because we don't want this worker to pass the entry to an output queue.
    return false;
}