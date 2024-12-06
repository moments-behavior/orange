/*
Adapted from Nvidia video sdk, FFmepgStreamer.
*/
#pragma once
extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
};
#include <iostream>
#include <fstream>
#include <thread>
#include "thread.h"
#include <cuda_runtime.h>

class FFmpegWriter
{
public:
    FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath, const char *metadata_file);
    ~FFmpegWriter();
    bool write_packet(uint8_t *pData, int nBytes, int nPts);
    void push_packet(uint8_t* pData, int nBytes, int nPts);
    void create_thread();
    void quit_thread();
    void join_thread();
    void write_one_pkt(AVPacket* pkt); 

private:
    AVFormatContext *oc = NULL;
    AVStream *vs = NULL;
    int nFps = 0;
    int nPts = 0;
    std::ofstream *metadata;
    lock_free_queue<AVPacket*> m_queue;
    std::thread m_thread;
    bool m_quitting;
    void write_thread();

    // CUDA memory management
    static const int NUM_CUDA_BUFFERS = 3;  // Triple buffering
    uint8_t* d_buffers[NUM_CUDA_BUFFERS];   // Device buffers
    int current_buffer = 0;
    size_t buffer_size = 0;
    cudaStream_t cuda_stream;
};
