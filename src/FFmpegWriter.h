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
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>

class FFmpegWriter
{
public:
    FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath, const char *metadata_file);
    ~FFmpegWriter();
    bool write_packet(uint8_t *pData, int nBytes, int nPts);
    void push_packet(uint8_t* pData, int nBytes, int nPts);
    AVPacket* pop_packet();
    void create_thread();
    void quit_thread()
    {
        m_quitting = true;
    };
    void join_thread()
    {
        m_thread.join();
    };
private:
    AVFormatContext *oc = NULL;
    AVStream *vs = NULL;
    int nFps = 0;
    int nPts = 0;
    std::ofstream *metadata;
    std::vector<AVPacket*> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::thread m_thread;
    bool m_quitting;
    void write_thread();
};
