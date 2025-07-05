#include "FFmpegWriter.h"
#include <cstdio>  // for FILE*
#include <cstring> // for memcpy

FFmpegWriter::FFmpegWriter(AVCodecID eCodecId, int nWidth, int nHeight,
                           int nFps, const char *szOutFilePath,
                           const char *metadata_file)
    : nFps(nFps), m_quitting(false) {
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

    vs->time_base = AVRational{1, nFps};

    // Everything is ready. Now open the output stream.
    output_file = fopen(szOutFilePath, "w+b");
    if (!output_file) {
        printf("FFMPEG: Failed to open %s for read/write\n", szOutFilePath);
        return;
    }

    // Create buffer for AVIOContext
    const int buffer_size = 4096;
    io_buffer = (uint8_t *)av_malloc(buffer_size);

    custom_io_ctx = avio_alloc_context(
        io_buffer, buffer_size,
        1, // write flag
        output_file,
        // Read callback
        [](void *opaque, uint8_t *buf, int buf_size) -> int {
            return fread(buf, 1, buf_size, (FILE *)opaque);
        },
        // Write callback
        [](void *opaque, uint8_t *buf, int buf_size) -> int {
            return fwrite(buf, 1, buf_size, (FILE *)opaque);
        },
        // Seek callback
        [](void *opaque, int64_t offset, int whence) -> int64_t {
            FILE *f = (FILE *)opaque;
            if (fseek(f, offset, whence) < 0)
                return -1;
            return ftell(f);
        });

    // Assign custom I/O context
    oc->pb = custom_io_ctx;
    oc->flags |= AVFMT_FLAG_CUSTOM_IO;

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "movflags", "faststart", 0);
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
}

FFmpegWriter::~FFmpegWriter() {
    // metadata->close();
    if (oc) {
        av_write_trailer(oc);
        avformat_free_context(oc);
    }

    if (output_file) {
        fflush(output_file);
        fclose(output_file);
    }
    if (custom_io_ctx) {
        av_free(custom_io_ctx->buffer);
        avio_context_free(&custom_io_ctx);
    }
}

void FFmpegWriter::push_packet(uint8_t *pData, int nBytes, int nPts) {
    AVPacket *pkt = av_packet_alloc();
    if (av_new_packet(pkt, nBytes) < 0) {
        std::cout << "Error, av_new_packet..." << std::endl;
        return;
    }
    memcpy(pkt->data, pData, nBytes);
    pkt->pts = av_rescale_q(nPts++, AVRational{1, nFps}, vs->time_base);
    // No B-frames
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;
    if (!memcmp(pData, "\x00\x00\x00\x01\x67", 5)) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    m_queue.push(pkt);
}

void FFmpegWriter::create_thread() {
    m_thread = std::thread(&FFmpegWriter::write_thread, this);
};

void FFmpegWriter::quit_thread() { m_quitting = true; };

void FFmpegWriter::join_thread() { m_thread.join(); };

void FFmpegWriter::write_one_pkt(AVPacket *pkt) {
    int ret = av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    if (ret < 0) {
        std::cout << "FFMPEG: Error while writing video frame" << std::endl;
    } else {
        av_packet_unref(pkt);
    }
}

// off-thread saving
void FFmpegWriter::write_thread() {
    while (!m_quitting) {
        std::shared_ptr<AVPacket *> pkt(m_queue.pop());
        if (pkt) {
            write_one_pkt(*pkt.get());
        }
    }

    // check if there is more in the queue
    while (true) {
        std::shared_ptr<AVPacket *> pkt(m_queue.pop());
        if (!pkt) {
            // empty queue
            break;
        } else {
            write_one_pkt(*pkt.get());
        }
    }
}

// this function only used for on thread saving
bool FFmpegWriter::write_packet(uint8_t *pData, int nBytes, int nPts) {

    // AVPacket pkt = {0};
    // av_init_packet(&pkt);
    AVPacket *pkt;
    pkt = av_packet_alloc();

    pkt->pts = av_rescale_q(nPts++, AVRational{1, nFps}, vs->time_base);
    // No B-frames
    pkt->dts = pkt->pts;
    pkt->stream_index = vs->index;
    pkt->data = pData;
    pkt->size = nBytes;

    // Look for a NAL unit type: H.264 (7 = SPS), HEVC (32 = VPS, 33 = SPS)
    if (nBytes >= 6 && !memcmp(pData, "\x00\x00\x00\x01", 4)) {
        uint8_t nal_unit_type = pData[4] & 0x1F; // H.264
        if (nal_unit_type == 5 || nal_unit_type == 7) {
            pkt->flags |= AV_PKT_FLAG_KEY;
        }

        // Try HEVC (NAL unit header is 2 bytes after start code)
        uint8_t hevc_nal_type = (pData[4] >> 1) & 0x3F; // HEVC NAL unit type
        if (hevc_nal_type == 19 || hevc_nal_type == 20 || hevc_nal_type == 32 ||
            hevc_nal_type == 33) {
            pkt->flags |= AV_PKT_FLAG_KEY;
        }
    }

    // Write the compressed frame into the output
    int ret = av_write_frame(oc, pkt);
    av_write_frame(oc, NULL);
    if (ret < 0) {
        std::cout << "FFMPEG: Error while writing video frame" << std::endl;
    }
    return true;
}
