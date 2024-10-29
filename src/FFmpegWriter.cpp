// #include "FFmpegWriter.h"

// FFmpegWriter::FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath, const char *metadata_file) : nFps(nFps), m_quitting(false)
// {
//     oc = avformat_alloc_context();
//     if (!oc) {
//         printf("FFMPEG: avformat_alloc_context error");
//         return;
//     }

//     // Set format on oc - changed from mpegts to mp4
//     AVOutputFormat *fmt = av_guess_format("mp4", NULL, NULL);
//     if (!fmt) {
//         printf("Invalid format");
//         return;
//     }
//     fmt->video_codec = eCodecId;
//     oc->oformat = fmt;

//     // Add video stream to oc
//     vs = avformat_new_stream(oc, NULL);
//     if (!vs) {
//         printf("FFMPEG: Could not alloc video stream");
//         return;
//     }
//     vs->id = 0;

//     // Set video parameters
//     AVCodecParameters *vpar = vs->codecpar;
//     vpar->codec_id = fmt->video_codec;
//     vpar->codec_type = AVMEDIA_TYPE_VIDEO;
//     vpar->width = nWidth;
//     vpar->height = nHeight;

//     // Set up stream timebase for MP4
//     vs->time_base = (AVRational){1, nFps};

//     // Everything is ready. Now open the output stream.
//     if (avio_open(&oc->pb, szOutFilePath, AVIO_FLAG_WRITE) < 0) {
//         printf("FFMPEG: Could not open %s", szOutFilePath);
//         return;
//     }

//     // Write the container header
//     if (avformat_write_header(oc, NULL)) {
//         printf("FFMPEG: avformat_write_header error!");
//         return;
//     }
// }

#include "FFmpegWriter.h"

FFmpegWriter::FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath, const char *metadata_file) 
    : nFps(nFps), m_quitting(false)
{
    oc = avformat_alloc_context();
    if (!oc) {
        printf("FFMPEG: avformat_alloc_context error");
        return;
    }

    // Set format on oc - use MP4 instead of MPEGTS
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

    // Set up stream timebase for MP4
    vs->time_base = (AVRational){1, nFps};

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

FFmpegWriter::~FFmpegWriter()
{
    if (oc) {
        av_write_trailer(oc);
        avio_close(oc->pb);
        avformat_free_context(oc);
    }
}

void FFmpegWriter::push_packet(uint8_t* pData, int nBytes, int nPts)
{
    AVPacket *pkt = av_packet_alloc();
    if (av_new_packet(pkt, nBytes) < 0) {
        std::cout << "Error, av_new_packet..." << std::endl;
        return;
    }
    memcpy(pkt->data, pData, nBytes);   
    pkt->pts = av_rescale_q(nPts++, AVRational{1, nFps}, vs->time_base);
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;    
    if(!memcmp(pData, "\x00\x00\x00\x01\x67", 5)) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    m_queue.push(pkt);
}

bool FFmpegWriter::write_packet(uint8_t *pData, int nBytes, int nPts)
{
    AVPacket *pkt = av_packet_alloc();
    pkt->pts = av_rescale_q(nPts++, AVRational{1, nFps}, vs->time_base);
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;
    pkt->data = pData;
    pkt->size = nBytes;

    if(!memcmp(pData, "\x00\x00\x00\x01\x67", 5)) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }

    int ret = av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    if (ret < 0) {
        std::cout << "FFMPEG: Error while writing video frame" << std::endl;
    }
    av_packet_free(&pkt);
    return true;
}

void FFmpegWriter::write_one_pkt(AVPacket* pkt)
{
    int ret = av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    if (ret < 0) {
        std::cout << "FFMPEG: Error while writing video frame" << std::endl;
    } else {
        av_packet_unref(pkt);
    }
}

void FFmpegWriter::create_thread()
{
    m_thread = std::thread(&FFmpegWriter::write_thread, this);
}

void FFmpegWriter::quit_thread()
{
    m_quitting = true;
}

void FFmpegWriter::join_thread()
{
    m_thread.join();
}

void FFmpegWriter::write_thread()
{
    while(!m_quitting) {
        std::shared_ptr<AVPacket*> pkt(m_queue.pop());
        if(pkt) {
            write_one_pkt(*pkt.get());
        }
    }   

    // check if there is more in the queue 
    while(true) {
        std::shared_ptr<AVPacket*> pkt(m_queue.pop());
        if (!pkt) {
            // empty queue
            break;
        } 
        else {
            write_one_pkt(*pkt.get());
        }
    }
}