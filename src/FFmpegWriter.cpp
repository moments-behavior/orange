#include "FFmpegWriter.h"


FFmpegWriter::FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath, const char *metadata_file) : nFps(nFps), m_quitting(false)
{
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
    
    metadata = new std::ofstream();
    metadata->open(metadata_file);
    if (!(*metadata))
    {
        std::cout << "File did not open!";
        return;
    }
    *metadata << "frame_id, keyframe\n";
}

FFmpegWriter::~FFmpegWriter()
{
    if (oc) {
        av_write_trailer(oc);
        avio_close(oc->pb);
        avformat_free_context(oc);
    }
    metadata->close();
}

void FFmpegWriter::push_packet(uint8_t* pData, int nBytes, int nPts)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    AVPacket *pkt = av_packet_alloc();
    if (av_new_packet(pkt, nBytes) < 0) {
        std::cout << "Error, av_new_packet..." << std::endl;
        return;
    }
    memcpy(pkt->data, pData, nBytes);   
    pkt->pts = av_rescale_q(nPts++, AVRational {1, nFps}, vs->time_base);
    // No B-frames
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;    
    if(!memcmp(pData, "\x00\x00\x00\x01\x67", 5)) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    m_queue.push_back(pkt);
    m_cond.notify_one();
}

AVPacket* FFmpegWriter::pop_packet()
{
    std::unique_lock<std::mutex> lock(m_mutex);

    // wait until queue is not empty 
    m_cond.wait(lock,
                [this]() { return !m_queue.empty(); });

    // retrieve item
    AVPacket* item = m_queue.front();
    m_queue.erase(m_queue.begin());
    // return item
    return item;
}

void FFmpegWriter::create_thread()
{
    m_thread = std::thread(&FFmpegWriter::write_thread, this);
};

// off-thread saving
void FFmpegWriter::write_thread()
{
    while(!m_quitting) {
        while(!m_queue.empty()) {
            AVPacket* pkt = pop_packet();
            int ret = av_write_frame(oc, pkt);
            av_write_frame(oc, NULL);
            if (ret < 0) {
                std::cout << "FFMPEG: Error while writing video frame" << std::endl;
            } else {
                av_packet_unref(pkt);
            }
        }
    }
}

// this function only used for on thread saving
bool FFmpegWriter::write_packet(uint8_t * pData, int nBytes, int nPts)
{

    //AVPacket pkt = {0};
    //av_init_packet(&pkt);
    AVPacket *pkt;
    pkt = av_packet_alloc();

    pkt->pts = av_rescale_q(nPts++, AVRational {1, nFps}, vs->time_base);
    // No B-frames
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;
    pkt->data = pData;
    pkt->size = nBytes;

    if(!memcmp(pData, "\x00\x00\x00\x01\x67", 5)) {
        pkt->flags |= AV_PKT_FLAG_KEY;
        *metadata << nPts << "," << 1 << std::endl;
    } else {
        *metadata << nPts << "," << 0 << std::endl;
    }

    // Write the compressed frame into the output
    int ret = av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    if (ret < 0) {
        std::cout << "FFMPEG: Error while writing video frame" << std::endl;

    }
    return true;
}