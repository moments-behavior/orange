#include "FFmpegWriter.h"


FFmpegWriter::FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight, int nFps, const char *szOutFilePath, const char *metadata_file) : nFps(nFps), m_quitting(false)
{
    oc = avformat_alloc_context();
    if (!oc) {
        printf("FFMPEG: avformat_alloc_context error");
        return;
    }

    // Set format on oc
    AVOutputFormat *fmt = av_guess_format("mp4", NULL, NULL);  // Changed from mpegts to mp4
    if (!fmt) {
        printf("Invalid format");
        return;
    }

    // Set codec to use NVENC
    if (eCodecId == AV_CODEC_ID_H264) {
        fmt->video_codec = AV_CODEC_ID_H264;
        const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
        if (!codec) {
            printf("Could not find h264_nvenc encoder\n");
            return;
        }
    } else if (eCodecId == AV_CODEC_ID_HEVC) {
        fmt->video_codec = AV_CODEC_ID_HEVC;
        const AVCodec *codec = avcodec_find_encoder_by_name("hevc_nvenc");
        if (!codec) {
            printf("Could not find hevc_nvenc encoder\n");
            return;
        }
    }
    
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
    
    // metadata = new std::ofstream();
    // metadata->open(metadata_file);
    // if (!(*metadata))
    // {
    //     std::cout << "File did not open!";
    //     return;
    // }
    // *metadata << "frame_id, keyframe\n";

    // Initialize CUDA buffers
    buffer_size = nWidth * nHeight * 4;  // Assuming 4 bytes per pixel
    cudaStreamCreate(&cuda_stream);
    
    for(int i = 0; i < NUM_CUDA_BUFFERS; i++) {
        cudaMalloc(&d_buffers[i], buffer_size);
    }
}

FFmpegWriter::~FFmpegWriter()
{
    if (oc) {
        av_write_trailer(oc);
        avio_close(oc->pb);
        avformat_free_context(oc);
    }
    
    // Cleanup CUDA resources
    cudaStreamSynchronize(cuda_stream);
    cudaStreamDestroy(cuda_stream);
    
    for(int i = 0; i < NUM_CUDA_BUFFERS; i++) {
        if(d_buffers[i]) {
            cudaFree(d_buffers[i]);
        }
    }
}

void FFmpegWriter::push_packet(uint8_t* pData, int nBytes, int nPts)
{
    // Copy to current GPU buffer
    cudaMemcpyAsync(d_buffers[current_buffer], pData, nBytes, cudaMemcpyDeviceToDevice, cuda_stream);
    
    AVPacket *pkt = av_packet_alloc();
    if (av_new_packet(pkt, nBytes) < 0) {
        std::cout << "Error, av_new_packet..." << std::endl;
        return;
    }
    
    // Copy from GPU buffer to packet
    cudaMemcpyAsync(pkt->data, d_buffers[current_buffer], nBytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);  // Ensure copy is complete
    
    current_buffer = (current_buffer + 1) % NUM_CUDA_BUFFERS;
    
    pkt->pts = av_rescale_q(nPts++, AVRational {1, nFps}, vs->time_base);
    // No B-frames
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;    
    if(!memcmp(pkt->data, "\x00\x00\x00\x01\x67", 5)) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    m_queue.push(pkt);
}

void FFmpegWriter::create_thread()
{
    m_thread = std::thread(&FFmpegWriter::write_thread, this);
};

void FFmpegWriter::quit_thread()
{
    m_quitting = true;
};

void FFmpegWriter::join_thread()
{
    m_thread.join();
};

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

// off-thread saving
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
        // *metadata << nPts << "," << 1 << std::endl;
    } else {
        // *metadata << nPts << "," << 0 << std::endl;
    }

    // Write the compressed frame into the output
    int ret = av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    if (ret < 0) {
        std::cout << "FFMPEG: Error while writing video frame" << std::endl;

    }
    return true;
}