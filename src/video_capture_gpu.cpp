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
    cudaMemset(debayer->d_debayer, 0xFF, size_pic);
    
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

static inline void duplicate_channel_gpu(CameraParams *camera_params, FrameGPU *frame_original, Debayer *debayer)
{
    const NppStatus npp_result = nppiDup_8u_C1AC4R(
        frame_original->d_orig,
        camera_params->width * sizeof(unsigned char),
        debayer->d_debayer,
        camera_params->width * sizeof(uchar4),
        debayer->size);

    if (npp_result != 0)
    {
        std::cout << "\nNPP error %d \n"
                  << npp_result << std::endl;
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

static inline void write_meatadata(ofstream *metadata, CameraEmergent *ecam)
{
    unsigned int offsetx; 
    unsigned int offsety; 
    EVT_CameraGetUInt32Param(&ecam->camera, "OffsetX", &offsetx);
    EVT_CameraGetUInt32Param(&ecam->camera, "OffsetY", &offsety);
    *metadata << ecam->frame_recv.frame_id << "," << ecam->frame_recv.timestamp << "," << offsetx << "," << offsety << endl;
}

static inline void PTP_timestamp_checking(PTPState *ptp_state, CameraEmergent *ecam, CameraState *camera_state)
{

    EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh", &ptp_state->ptp_time_high);
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow", &ptp_state->ptp_time_low);

    ptp_state->ptp_time = (((unsigned long long)(ptp_state->ptp_time_high)) << 32) | ((unsigned long long)(ptp_state->ptp_time_low));
    ptp_state->frame_ts = ecam->frame_recv.timestamp;
    // printf("camera %d, framecount %d, timestamp %f ms \n", camera_params.camera_id, frame_count, frame_ts * 1e-6);

    if (camera_state->frame_count != 0)
    {
        ptp_state->ptp_time_delta = ptp_state->ptp_time - ptp_state->ptp_time_prev;
        ptp_state->ptp_time_delta_sum += ptp_state->ptp_time_delta;

        ptp_state->frame_ts_delta = ptp_state->frame_ts - ptp_state->frame_ts_prev;
        ptp_state->frame_ts_delta_sum += ptp_state->frame_ts_delta;
    }

    ptp_state->ptp_time_prev = ptp_state->ptp_time;
    ptp_state->frame_ts_prev = ptp_state->frame_ts;
}


static inline void get_one_frame(CameraState *camera_state, CameraControl *camera_control, CameraEmergent *ecam, CameraParams *camera_params, Debayer *debayer, FrameGPU *frame_original, EncoderContext *encoder, Writer *writer, PTPState *ptp_state)
{
    camera_state->camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
    if (camera_control->sync_camera) {
        PTP_timestamp_checking(ptp_state, ecam, camera_state);
    }

    if (!camera_state->camera_return)
    {
        // Counting dropped frames through frame_id as redundant check.
        if (((ecam->frame_recv.frame_id) != camera_state->id_prev + 1) && (camera_state->frame_count != 0))
            camera_state->dropped_frames++;
        else
        {
            camera_state->frames_recd++;
            upload_frame_to_gpu(camera_params, frame_original, ecam);
            if (camera_params->color){
                debayer_frame_gpu(camera_params, frame_original, debayer);
            } else {
                duplicate_channel_gpu(camera_params, frame_original, debayer);
            }
            if (camera_control->record_video)
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
    while (camera_control->subscribe)
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
    *frame_metadata << "frame_id,timestamp,offsetx,offsety\n";
}

static inline void initialize_writer(Writer *writer, CameraParams *camera_params, string folder_name)
{
    writer->video_file = folder_name + "/Cam" + camera_params->camera_name + ".mp4";
    writer->metadata_file = folder_name + "/Cam" + camera_params->camera_name + "_meta.csv";
    writer->video = new FFmpegWriter(AV_CODEC_ID_H264, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str());
    writer->metadata = new ofstream();
    open_metadata_file(writer->metadata, writer->metadata_file);
}

static inline void close_writer(EncoderContext *encoder, Writer *writer)
{
    encoder->pEnc->EndEncode(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket)
    {
        writer->video->Write(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
    }
    encoder->pEnc->DestroyEncoder();
    (*writer->metadata).close();
}

static inline void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam)
{
    // Show raw offsets.
    for (unsigned int i = 0; i < 5;)
    {
        EVT_CameraGetInt32Param(&ecam->camera, "PtpOffset", &ptp_state->ptp_offset);
        if (ptp_state->ptp_offset != ptp_state->ptp_offset_prev)
        {
            ptp_state->ptp_offset_sum += ptp_state->ptp_offset;
            i++;
            // printf("Offset %d: %d\n", i, ptp_offset);
        }
        ptp_state->ptp_offset_prev = ptp_state->ptp_offset;
    }
    printf("Offset Average: %d\n", ptp_state->ptp_offset_sum / 5);
}

static inline void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params, CameraParams *camera_params, CameraEmergent *ecam, unsigned int delay_in_second)
{

    if (ptp_params->ptp_counter == camera_params->num_cameras - 1)
    {
        ptp_state->ptp_time = get_current_PTP_time(&ecam->camera);
        ptp_params->ptp_global_time = ((unsigned long long)delay_in_second) * 1000000000 + ptp_state->ptp_time;
    }
    uint64_t ptp_counter = sync_fetch_and_add(&ptp_params->ptp_counter, 1);
    printf("%lu\n", ptp_counter);
    while (ptp_params->ptp_counter != camera_params->num_cameras)
    {
        printf(".");
        fflush(stdout);
    }

    unsigned long long ptp_time_plus_delta_to_start = ptp_params->ptp_global_time;
    ptp_state->ptp_time_plus_delta_to_start_low = (unsigned int)(ptp_time_plus_delta_to_start & 0xFFFFFFFF);
    ptp_state->ptp_time_plus_delta_to_start_high = (unsigned int)(ptp_time_plus_delta_to_start >> 32);
    EVT_CameraSetUInt32Param(&ecam->camera, "PtpAcquisitionGateTimeHigh", ptp_state->ptp_time_plus_delta_to_start_high);
    EVT_CameraSetUInt32Param(&ecam->camera, "PtpAcquisitionGateTimeLow", ptp_state->ptp_time_plus_delta_to_start_low);
    ptp_state->ptp_time_plus_delta_to_start_uint = ptp_time_plus_delta_to_start;
    ptp_state->ptp_time_plus_delta_to_start = ptp_params->ptp_global_time;
    printf("PTP Gate time(ns): %llu\n", ptp_time_plus_delta_to_start);
}

static inline void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam)
{
    printf("Grabbing Frames after countdown...\n");
    ptp_state->ptp_time_countdown = 0;
    // Countdown code
    do
    {
        EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
        EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh", &ptp_state->ptp_time_high);
        EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow", &ptp_state->ptp_time_low);
        ptp_state->ptp_time = (((unsigned long long)(ptp_state->ptp_time_high)) << 32) | ((unsigned long long)(ptp_state->ptp_time_low));

        if (ptp_state->ptp_time > ptp_state->ptp_time_countdown)
        {
            printf("%llu\n", (ptp_state->ptp_time_plus_delta_to_start - ptp_state->ptp_time) / 1000000000);
            ptp_state->ptp_time_countdown = ptp_state->ptp_time + 1000000000; // 1s
        }

    } while (ptp_state->ptp_time <= ptp_state->ptp_time_plus_delta_to_start);
    // Countdown done.
    printf("\n");
}


void aquire_frames_gpu(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_control, unsigned char *display_buffer, string encoder_setup, string folder_name, PTPParams *ptp_params)
{
    ck(cudaSetDevice(camera_params->gpu_id));

    CameraState camera_state;
    FrameGPU frame_original;
    initalize_gpu_frame(&frame_original, camera_params);
    Debayer debayer;
    initialize_gpu_debayer(&debayer, camera_params);

    cudaStream_t stream1;
    std::thread t_stream;
    if (camera_control->stream) {
        ck(cudaStreamCreate(&stream1));
        t_stream = std::thread(&copy_to_display_buffer, camera_params, camera_control, display_buffer, &debayer, stream1);
    }

    // encoding
    EncoderContext encoder;
    Writer writer;
    PTPState ptp_state;

    if (camera_control->record_video) {
        initialize_encoder(&encoder, encoder_setup, camera_params);
        initialize_writer(&writer, camera_params, folder_name);
    }

    if (camera_control->sync_camera) {
        show_ptp_offset(&ptp_state, ecam);
        start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3);
    }
    
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"));
    if (camera_control->sync_camera) {
        grab_frames_after_countdown(&ptp_state, ecam);
    }

    StopWatch w;
    w.Start();

    // int OFFSET_X_VAL = 2848;
    // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetX", OFFSET_X_VAL);
    // int offset = 0;
    // int phase = 1;

    while (camera_control->subscribe)
    {
        // int OFFSET_Y_VAL = 1300 + offset * 4;
        // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetY", OFFSET_Y_VAL);

        get_one_frame(&camera_state, camera_control, ecam, camera_params, &debayer, &frame_original, &encoder, &writer, &ptp_state);

        // if (offset == 200) {
        //     phase = -1;
        // }
        // if (offset == 0) {
        //     phase = 1;
        // }
        // if (phase == -1) {
        //     offset--;
        // } else { offset++; }
    }
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"));

    if (camera_control->record_video) {
        close_writer(&encoder, &writer);
    }
    double time_diff = w.Stop();
    report_statistics(camera_params, &camera_state, time_diff);

    if (camera_control->stream) {
        if (t_stream.joinable())
            t_stream.join();
    }
    cudaStreamDestroy(stream1);
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    delete writer.video;
    delete writer.metadata;
    delete encoder.pEnc;
}