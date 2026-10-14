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
    // is_keyframe (optional) reports whether the packet was flagged as a key
    // frame (IDR / parameter sets) in the container.
    bool write_packet(uint8_t *pData, int nBytes, int nPts,
                      bool *is_keyframe = nullptr);

  private:
    AVFormatContext *oc = NULL;
    AVStream *vs = NULL;
    int nFps = 0;
    std::ofstream *metadata;
};
