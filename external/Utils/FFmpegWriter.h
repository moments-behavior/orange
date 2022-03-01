/*
Adapted from Nvidia video sdk, FFmepgStreamer.
*/

#pragma once
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
};


class FFmpegWriter {
private:
    AVFormatContext *oc = NULL;
    AVStream *vs = NULL;
    int nFps = 0;

public:
    FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath) : nFps(nFps) {
        oc = avformat_alloc_context();
        if (!oc) {
            printf("FFMPEG: avformat_alloc_context error");
            return;
        }

        // Set format on oc
        AVOutputFormat *fmt = av_guess_format("mpegts", NULL, NULL);
        if (!fmt) {
            printf("Invalid format");
            return;
        }
        fmt->video_codec = eCodecId;
        oc->oformat = fmt;

        // Add video stream to oc
        vs = avformat_new_stream(oc, NULL);
        if (!vs) {
            printf("FFMPEG: Could not alloc video stream");
            return;
        }
        vs->id = 0;

        // Set video parameters
        AVCodecParameters *vpar = vs->codecpar;
        vpar->codec_id = fmt->video_codec;
        vpar->codec_type = AVMEDIA_TYPE_VIDEO;
        vpar->width = nWidth;
        vpar->height = nHeight;

        // Everything is ready. Now open the output stream.
        if (avio_open(&oc->pb, szOutFilePath, AVIO_FLAG_WRITE) < 0) {
            printf("FFMPEG: Could not open %s", szOutFilePath);
            return;
        }

        // Write the container header
        if (avformat_write_header(oc, NULL)) {
            printf("FFMPEG: avformat_write_header error!");
            return;
        }
    }
    ~FFmpegWriter() {
        if (oc) {
            av_write_trailer(oc);
            avio_close(oc->pb);
            avformat_free_context(oc);
        }
    }

    bool Write(uint8_t *pData, int nBytes, int nPts) {
        AVPacket pkt = {0};
        av_init_packet(&pkt);
        pkt.pts = av_rescale_q(nPts++, AVRational {1, nFps}, vs->time_base);
        // No B-frames
        pkt.dts = pkt.pts;
        pkt.stream_index = vs->index;
        pkt.data = pData;
        pkt.size = nBytes;

        if(!memcmp(pData, "\x00\x00\x00\x01\x67", 5)) {
            pkt.flags |= AV_PKT_FLAG_KEY;
        }

        // Write the compressed frame into the output
        int ret = av_write_frame(oc, &pkt);
        av_write_frame(oc, NULL);
        if (ret < 0) {
            // LOG(ERROR) << "FFMPEG: Error while writing video frame";
            std::cout << "FFMPEG: Error while writing video frame" << std::endl;

        }
        return true;
    }
};
