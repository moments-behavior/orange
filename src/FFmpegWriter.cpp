#include "FFmpegWriter.h"
#include "libavutil/rational.h"
#include <cstdio>  // for FILE*
#include <cstring> // for memcpy
#include <iostream>

FFmpegWriter::FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight,
                           int nFps, const char *szOutFilePath,
                           const char *metadata_file)
    : nFps(nFps) {
    oc = avformat_alloc_context();
    if (!oc) {
        printf("FFMPEG: avformat_alloc_context error");
        return;
    }

    // Set format on oc
    AVOutputFormat *fmt = av_guess_format("mp4", NULL, NULL);
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
    // Set correct codec_tag for Apple compatibility
    if (vpar->codec_id == AV_CODEC_ID_H264) {
        vpar->codec_tag = MKTAG('a', 'v', 'c', '1');
    } else if (vpar->codec_id == AV_CODEC_ID_HEVC) {
        vpar->codec_tag = MKTAG('h', 'v', 'c', '1');
    }

    vs->time_base = AVRational{1, nFps};

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

    metadata = new std::ofstream();
    metadata->open(metadata_file);
    if (!(*metadata)) {
        std::cout << "File did not open!";
        return;
    }
    *metadata << "frame_id, keyframe\n";
}

FFmpegWriter::~FFmpegWriter() {
    if (metadata) {
        metadata->close();
        delete metadata;
    }

    if (oc) {
        av_write_trailer(oc);
        avio_close(oc->pb);
        avformat_free_context(oc);
    }
}

bool FFmpegWriter::write_packet(uint8_t *pData, int nBytes, int nPts) {
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        std::cerr << "Failed to allocate AVPacket\n";
        return false;
    }

    pkt->data = pData;
    pkt->size = nBytes;
    pkt->pts = av_rescale_q(nPts, AVRational{1, nFps}, vs->time_base);
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;

    bool is_keyframe = false;

    if (nBytes >= 6 && !memcmp(pData, "\x00\x00\x00\x01", 4)) {
        uint8_t nal_unit_type = pData[4] & 0x1F; // H.264
        if (nal_unit_type == 5 || nal_unit_type == 7)
            is_keyframe = true;

        uint8_t hevc_nal_type = (pData[4] >> 1) & 0x3F; // HEVC
        if (hevc_nal_type == 19 || hevc_nal_type == 20 || hevc_nal_type == 32 ||
            hevc_nal_type == 33)
            is_keyframe = true;
    }

    if (is_keyframe)
        pkt->flags |= AV_PKT_FLAG_KEY;

    if (metadata && metadata->is_open()) {
        *metadata << nPts << "," << (is_keyframe ? 1 : 0) << "\n";
    }

    int ret = av_write_frame(oc, pkt);
    if (ret < 0) {
        std::cerr << "FFMPEG: Error while writing frame\n";
    }

    av_packet_unref(pkt);
    av_packet_free(&pkt);
    return true;
}
