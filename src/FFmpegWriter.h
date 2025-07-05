/*
Adapted from Nvidia video sdk, FFmepgStreamer.
*/
#pragma once
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
};
#include <fstream>

class FFmpegWriter {
  public:
    FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps,
                 const char *szOutFilePath, const char *metadata_file);
    ~FFmpegWriter();
    bool write_packet(uint8_t *pData, int nBytes, int nPts);

  private:
    AVFormatContext *oc = NULL;
    AVStream *vs = NULL;
    int nFps = 0;
    std::ofstream *metadata;
};
