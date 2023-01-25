#include "video_capture_gpu.h"
#include "cuda_line_reorder.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
}

static inline void upload_frame_to_gpu(CameraParams *camera_params, FrameGPU *frame_original, CameraEmergent *ecam)
{
    if (camera_params->need_reorder)
    {
        if (camera_params->gpu_direct)
        {
            // line reorder using gpu
            GSPRINT4521_Convert(frame_original->d_orig, (const unsigned char *)ecam->frame_recv.imagePtr,
                                camera_params->width, camera_params->height, camera_params->width, camera_params->width, 0); // only for  8 bit
        }
        else
        {
            EVT_FrameConvert(&ecam->frame_recv, &ecam->frame_reorder, 0, 0, ecam->camera.linesReorderHandle);
            // upload to gpu, can consider do this in a different thread, write encoder as a callback function?
            ck(cudaMemcpy(frame_original->d_orig, &ecam->frame_reorder.imagePtr, frame_original->size_pic, cudaMemcpyHostToDevice));
        }
    }
    else
    {
        if (camera_params->gpu_direct)
        {
            ck(cudaMemcpy(frame_original->d_orig, ecam->frame_recv.imagePtr, frame_original->size_pic, cudaMemcpyDeviceToDevice));
        }
        else
        {
            ck(cudaMemcpy(frame_original->d_orig, ecam->frame_recv.imagePtr, frame_original->size_pic, cudaMemcpyHostToDevice));
        }
    }
}

static inline void initialize_gpu_debayer(Debayer *debayer, CameraParams *camera_params)
{
    int output_channels = 4;
    int size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char) * output_channels;
    cudaMalloc((void **)&debayer->d_debayer, size_pic);
    debayer->size.width = camera_params->width;
    debayer->size.height = camera_params->height;
    debayer->nAlpha = 255;
    debayer->roi.x = 0;
    debayer->roi.y = 0;
    debayer->roi.width = camera_params->width;
    debayer->roi.height = camera_params->height;
    if (camera_params->need_reorder)
    {
        debayer->grid = NPPI_BAYER_GRBG;
    }
    else
    {
        debayer->grid = NPPI_BAYER_RGGB;
    }
}

static inline void debayer_frame_gpu(CameraParams *camera_params, FrameGPU *frame_original, Debayer *debayer)
{
    const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(frame_original->d_orig,
                                                         camera_params->width * sizeof(unsigned char),
                                                         debayer->size,
                                                         debayer->roi,
                                                         debayer->d_debayer,
                                                         camera_params->width * sizeof(uchar4),
                                                         debayer->grid,
                                                         NPPI_INTER_UNDEFINED,
                                                         debayer->nAlpha);
    if (npp_result != 0)
    {
        std::cout << "\nNPP error %d \n"
                  << npp_result << std::endl;
    }
}

static inline void encode_frame(EncoderContext *encoder, FFmpegWriter *writer, Debayer *debayer)
{
    // encoding
    const NvEncInputFrame *encoderInputFrame = encoder->pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(encoder->cuContext,
                                     debayer->d_debayer,
                                     0,
                                     (CUdeviceptr)encoderInputFrame->inputPtr,
                                     (int)encoderInputFrame->pitch,
                                     encoder->pEnc->GetEncodeWidth(),
                                     encoder->pEnc->GetEncodeHeight(),
                                     CU_MEMORYTYPE_DEVICE,
                                     encoderInputFrame->bufferFormat,
                                     encoderInputFrame->chromaOffsets,
                                     encoderInputFrame->numChromaPlanes);
    encoder->pEnc->EncodeFrame(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket)
    {
        // For each encoded packet
        writer->Write(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
    }
}

static inline void get_one_frame(CameraState *camera_state, CameraEmergent *ecam, CameraParams *camera_params, Debayer *debayer, FrameGPU *frame_original)
{
    camera_state->camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
    if (!camera_state->camera_return)
    {
        // Counting dropped frames through frame_id as redundant check.
        if (((ecam->frame_recv.frame_id) != camera_state->id_prev + 1) && (camera_state->frame_count != 0))
            camera_state->dropped_frames++;
        else
        {
            camera_state->frames_recd++;
            upload_frame_to_gpu(camera_params, frame_original, ecam);
            debayer_frame_gpu(camera_params, frame_original, debayer);
        }

        // In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if (ecam->frame_recv.frame_id == 65535)
            camera_state->id_prev = 0;
        else
            camera_state->id_prev = ecam->frame_recv.frame_id;

        camera_state->camera_return = EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv); // Re-queue.
        if (camera_state->camera_return)
            std::cout << "EVT_CameraQueueFrame Error!" << std::endl;

        if (camera_state->frame_count % 100 == 99)
        {
            printf(".");
            fflush(stdout);
        }
        if (camera_state->frame_count % 10000 == 9999)
            printf("\n");

        camera_state->frame_count++;
    }
    else
    {
        camera_state->dropped_frames++;
        std::cout << "EVT_CameraGetFrame Error" << camera_state->camera_return << std::endl;
    }
}

static inline void write_meatadata(ofstream *metadata, CameraEmergent *ecam)
{
    *metadata << ecam->frame_recv.frame_id << "," << ecam->frame_recv.timestamp << endl;
}

static inline void get_one_frame_encode(CameraState *camera_state, CameraControl *camera_control, CameraEmergent *ecam, CameraParams *camera_params, Debayer *debayer, FrameGPU *frame_original, EncoderContext *encoder, Writer *writer)
{
    camera_state->camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
    if (!camera_state->camera_return)
    {
        // Counting dropped frames through frame_id as redundant check.
        if (((ecam->frame_recv.frame_id) != camera_state->id_prev + 1) && (camera_state->frame_count != 0))
            camera_state->dropped_frames++;
        else
        {
            camera_state->frames_recd++;
            upload_frame_to_gpu(camera_params, frame_original, ecam);
            debayer_frame_gpu(camera_params, frame_original, debayer);
            if (!camera_control->pause_recording)
            {
                encode_frame(encoder, writer->video, debayer);
                write_meatadata(writer->metadata, ecam);
            }
        }

        // In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if (ecam->frame_recv.frame_id == 65535)
            camera_state->id_prev = 0;
        else
            camera_state->id_prev = ecam->frame_recv.frame_id;

        camera_state->camera_return = EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv); // Re-queue.
        if (camera_state->camera_return)
            std::cout << "EVT_CameraQueueFrame Error!" << std::endl;

        if (camera_state->frame_count % 100 == 99)
        {
            printf(".");
            fflush(stdout);
        }
        if (camera_state->frame_count % 10000 == 9999)
            printf("\n");

        camera_state->frame_count++;
    }
    else
    {
        camera_state->dropped_frames++;
        std::cout << "EVT_CameraGetFrame Error" << camera_state->camera_return << std::endl;
    }
}

static inline void report_statistics(CameraParams *camera_params, CameraState *camera_state, double time_diff)
{
    printf("\n");
    printf("Camera id: \t%d\n", camera_params->camera_id);
    printf("Frame count: \t%d\n", camera_state->frame_count);
    printf("Frame received: \t%d\n", camera_state->frames_recd);
    printf("Dropped Frames: \t%d\n", camera_state->dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", camera_state->frames_recd / time_diff);
}

static inline void copy_to_display_buffer(CameraParams *camera_params, CameraControl *camera_control, unsigned char *display_buffer, Debayer *debayer, cudaStream_t stream1)
{
    std::chrono::steady_clock::time_point steady_start, steady_end;
    while (camera_control->streaming)
    {
        steady_end = std::chrono::steady_clock::now();
        float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
        if (time_sec >= 0.03)
        {
            // ck(cudaMemcpy2D(display_buffer, camera_params->width*4, d_debayer, camera_params->width*4, camera_params->width*4, camera_params->height, cudaMemcpyDeviceToDevice));
            ck(cudaMemcpy2DAsync(display_buffer, camera_params->width * 4, debayer->d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice, stream1));
            steady_start = steady_end;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

static inline void initalize_gpu_frame(FrameGPU *frame_original, CameraParams *camera_params)
{
    frame_original->size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char);
    cudaMalloc((void **)&frame_original->d_orig, frame_original->size_pic);
}

// gpu pipeline, raw bayer images as input
void aquire_frames_gpu(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_control, unsigned char *display_buffer)
{
    ck(cudaSetDevice(camera_params->gpu_id));

    CameraState camera_state;
    FrameGPU frame_original;
    initalize_gpu_frame(&frame_original, camera_params);
    Debayer debayer;
    initialize_gpu_debayer(&debayer, camera_params);

    cudaStream_t stream1;
    ck(cudaStreamCreate(&stream1));
    std::thread t_stream = std::thread(&copy_to_display_buffer, camera_params, camera_control, display_buffer, &debayer, stream1);

    StopWatch w;
    w.Start();
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"));
    while (camera_control->streaming)
    {
        get_one_frame(&camera_state, ecam, camera_params, &debayer, &frame_original);
    }
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"));
    double time_diff = w.Stop();

    report_statistics(camera_params, &camera_state, time_diff);
    if (t_stream.joinable())
        t_stream.join();
    cudaStreamDestroy(stream1);
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
}

static inline void initialize_encoder(EncoderContext *encoder, string encoder_str, CameraParams *camera_params)
{
    encoder->eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    encoder->encodeCLIOptions = NvEncoderInitParam(encoder_str.c_str());
    CUdevice cuDevice;
    ck(cuDeviceGet(&cuDevice, camera_params->gpu_id));
    encoder->cuContext = NULL;
    ck(cuCtxCreate(&encoder->cuContext, 0, cuDevice));
    encoder->pEnc = new NvEncoderCuda(encoder->cuContext, camera_params->width, camera_params->height, encoder->eFormat);
    InitializeEncoder(encoder->pEnc, encoder->encodeCLIOptions, encoder->eFormat);
}

static inline void open_metadata_file(ofstream *frame_metadata, string metadata_file)
{
    frame_metadata->open(metadata_file.c_str());

    if (!(*frame_metadata))
    {
        std::cout << "File did not open!";
        return;
    }
    *frame_metadata << "frame_id,timestamp\n";
}

static inline void initialize_writer(Writer *writer, CameraParams *camera_params, string folder_name)
{
    writer->video_file = folder_name + "/Cam" + camera_params->camera_name + ".mp4";
    writer->metadata_file = folder_name + "/Cam" + camera_params->camera_name + "_meta.csv";
    writer->video = new FFmpegWriter(AV_CODEC_ID_H264, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str());
    writer->metadata = new ofstream();
    open_metadata_file(writer->metadata, writer->metadata_file);
}

void static inline close_writer(EncoderContext *encoder, Writer *writer)
{
    encoder->pEnc->EndEncode(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket)
    {
        writer->video->Write(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
    }
    encoder->pEnc->DestroyEncoder();
    (*writer->metadata).close();
}

void aquire_frames_gpu_encode(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_control, unsigned char *display_buffer, string encoder_setup, string folder_name)
{
    ck(cudaSetDevice(camera_params->gpu_id));

    CameraState camera_state;
    FrameGPU frame_original;
    initalize_gpu_frame(&frame_original, camera_params);
    Debayer debayer;
    initialize_gpu_debayer(&debayer, camera_params);

    cudaStream_t stream1;
    ck(cudaStreamCreate(&stream1));
    std::thread t_stream = std::thread(&copy_to_display_buffer, camera_params, camera_control, display_buffer, &debayer, stream1);

    // encoding
    EncoderContext encoder;
    initialize_encoder(&encoder, encoder_setup, camera_params);

    Writer writer;
    initialize_writer(&writer, camera_params, folder_name);

    StopWatch w;
    w.Start();
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"));
    while (camera_control->streaming)
    {
        get_one_frame_encode(&camera_state, camera_control, ecam, camera_params, &debayer, &frame_original, &encoder, &writer);
    }
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"));

    close_writer(&encoder, &writer);
    double time_diff = w.Stop();
    report_statistics(camera_params, &camera_state, time_diff);

    // cleanup
    if (t_stream.joinable())
        t_stream.join();
    cudaStreamDestroy(stream1);
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    delete writer.video;
    delete writer.metadata;
    delete encoder.pEnc;
}