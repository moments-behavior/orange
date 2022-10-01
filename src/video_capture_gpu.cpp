#include "video_capture_gpu.h"
#include "FFmpegWriter.h"
#include <iostream>
#include <fstream>
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


// gpu pipeline, raw bayer images as input
void aquire_frames_gpu_encode(Emergent::CEmergentCamera *camera, Emergent::CEmergentFrame *frame_recv, CameraParams* camera_params, const char *encoder_str, int* key_num_ptr, PTPParams* ptp_params, string folder_name, unsigned char* display_buffer, bool* encode_flag, bool* capture_pause)
{
    int camera_return{0};

    // unsigned int size_of_buffer;
    // size_of_buffer = frame_recv->CalculateBufferSize();

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;

    ck(cudaSetDevice(camera_params->gpu_id));
    
    // modularize these parts: 1. debayer; 2. encoding; 
    // gpu: upload raw images and color debayer
    int output_channels = 4;
    unsigned char *d_orig;
    int size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char);
    cudaMalloc((void **)&d_orig, size_pic);
    unsigned char *d_debayer;
    cudaMalloc((void **)&d_debayer, size_pic * output_channels);

    // for encoding purpose 
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    NvEncoderInitParam encodeCLIOptions = NvEncoderInitParam(encoder_str);
    ck(cuInit(0));
    CUdevice cuDevice = 0;

    // specify which gpu
    ck(cuDeviceGet(&cuDevice, camera_params->gpu_id));

    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
    //ck(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, camera_params->width, camera_params->height, eFormat));
    

    // debayer
    NppiSize size;
    size.width = camera_params->width;
    size.height = camera_params->height;
    Npp8u nAlpha = 255;

    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = camera_params->width;
    roi.height = camera_params->height;

    NppiBayerGridPosition grid;
    
    if(camera_params->need_reorder)
    {
        grid = NPPI_BAYER_GRBG;
    }
    else{
        grid = NPPI_BAYER_RGGB;
    }
    

    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

    // For receiving encoded packets
    std::vector<std::vector<uint8_t>> vPacket;
    int num_frame_encode = 0;

    // for writing 

    string video_file = folder_name + "/Cam" + std::to_string(camera_params->camera_id) + ".mp4";
    const char *output_file = video_file.c_str();
    FFmpegWriter writer(AV_CODEC_ID_H264, camera_params->width, camera_params->height, camera_params->frame_rate, output_file);

    string metadata_file = folder_name + "/Cam" + std::to_string(camera_params->camera_id) + "_meta.csv";
    ofstream frame_metadata;
    frame_metadata.open(metadata_file.c_str());
    if(!frame_metadata)
    {
        std::cout << "File did not open!";
        exit(1);
    }
    frame_metadata << "frame_id,timestamp\n";
    

    Emergent::CEmergentFrame FrameConvert;
    //Five params used for memory allocation for converted frames. Worst case covers all models so no recompilation required.
    FrameConvert.size_x = 5120;
    FrameConvert.size_y = 4096;
    FrameConvert.pixel_type = GVSP_PIX_BAYGB8;
    FrameConvert.convertColor = EVT_COLOR_CONVERT_NONE;
    FrameConvert.convertBitDepth = EVT_CONVERT_NONE;
    EVT_AllocateFrameBuffer(camera, &FrameConvert, EVT_FRAME_BUFFER_DEFAULT);
    
    //*************************************PTP**************************************************
    // handel sync using 
    int ptp_offset, ptp_offset_sum=0, ptp_offset_prev=0;
    unsigned int ptp_time_low, ptp_time_high, ptp_time_plus_delta_to_start_low, ptp_time_plus_delta_to_start_high;
    unsigned long long ptp_time_delta_sum = 0, ptp_time_delta, ptp_time, ptp_time_prev, ptp_time_countdown;
    unsigned long long frame_ts; 
    unsigned long long frame_ts_prev, frame_ts_delta, frame_ts_delta_sum = 0;
    char ptp_status[100];
    unsigned long ptp_status_sz_ret;
    

    //Show raw offsets.
    for (unsigned int i = 0; i < 5;)
    {
        EVT_CameraGetInt32Param(camera, "PtpOffset", &ptp_offset);
        if (ptp_offset != ptp_offset_prev)
        {
            ptp_offset_sum += ptp_offset;
            i++;
            //printf("Offset %d: %d\n", i, ptp_offset);
        }
        ptp_offset_prev = ptp_offset;
    }
    printf("Offset Average: %d\n", ptp_offset_sum / 5);


    if(ptp_params->ptp_counter == camera_params->num_cameras -1)
    {
        ptp_time = get_current_PTP_time(camera);
        unsigned int ptp_time_plus_delta_to_start_uint = 10;
        ptp_params->ptp_global_time = ((unsigned long long)ptp_time_plus_delta_to_start_uint) * 1000000000 + ptp_time;
    }
    uint64_t ptp_counter = sync_fetch_and_add(&ptp_params->ptp_counter, 1);
    printf("%lu\n", ptp_counter);
    while(ptp_params->ptp_counter != camera_params->num_cameras)
    {
        printf(".");
        fflush(stdout);
    }
    

    unsigned long long ptp_time_plus_delta_to_start = ptp_params->ptp_global_time;
    ptp_time_plus_delta_to_start_low  = (unsigned int)(ptp_time_plus_delta_to_start & 0xFFFFFFFF);
    ptp_time_plus_delta_to_start_high = (unsigned int)(ptp_time_plus_delta_to_start >> 32);
    EVT_CameraSetUInt32Param(camera, "PtpAcquisitionGateTimeHigh", ptp_time_plus_delta_to_start_high);
    EVT_CameraSetUInt32Param(camera, "PtpAcquisitionGateTimeLow", ptp_time_plus_delta_to_start_low);
    printf("PTP Gate time(ns): %llu\n", ptp_time_plus_delta_to_start);
    
    // std::time_t ptp_info_time = (ptp_time_plus_delta_to_start/100000000push_back
    //*************************************Streaming**************************************************
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));
    
    
    printf("Grabbing Frames after countdown...\n");
    ptp_time_countdown = 0;
    //Countdown code
    do {
        EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch");
        EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high);
        EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low);
        ptp_time = (((unsigned long long)(ptp_time_high)) << 32) | ((unsigned long long)(ptp_time_low));

        if (ptp_time > ptp_time_countdown)
        {
            printf("%llu\n", (ptp_time_plus_delta_to_start - ptp_time) / 1000000000);
            ptp_time_countdown = ptp_time + 1000000000; //1s
        }

    } while (ptp_time <= ptp_time_plus_delta_to_start);
    //Countdown done.
    printf("\n");
    //********************************************************************************************

    StopWatch w;
    w.Start();
    int frame_count = 0;
    while (*key_num_ptr != 27)
    {
        // pause capture 
        while (*capture_pause) {
            // if the next frame hasn't been displayed, the queue is full, sleep  
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }        


        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        // printf("get frame\n");
       //////////////////////////////PTP timestamp checking////////////////////////////////////
        EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch");
        EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high);
        EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low);

        ptp_time = (((unsigned long long)(ptp_time_high)) << 32) | ((unsigned long long)(ptp_time_low));
        frame_ts = frame_recv->timestamp;
        // printf("camera %d, framecount %d, timestamp %f ms \n", camera_params.camera_id, frame_count, frame_ts * 1e-6);


        if (frame_count != 0)
        {
            ptp_time_delta = ptp_time - ptp_time_prev;
            ptp_time_delta_sum += ptp_time_delta;

            frame_ts_delta = frame_ts - frame_ts_prev;
            frame_ts_delta_sum += frame_ts_delta;
        }

        ptp_time_prev = ptp_time;
        frame_ts_prev = frame_ts;
    //     //////////////////////////////PTP timestamp checking////////////////////////////////////

        if (!camera_return)
        {
            //Counting dropped frames through frame_id as redundant check.
            if (((frame_recv->frame_id) != id_prev + 1) && (frame_count != 0))
                dropped_frames++;
            else
            {

                //frame_metadata << "frame_id " << frame_recv->frame_id << ", timestamp " << frame_ts << endl;                
                frame_metadata << frame_recv->frame_id << "," << frame_ts << endl;                
                frames_recd++;

                if(camera_params->need_reorder){
                    if (camera_params->gpu_direct){
                        // line reorder using gpu 
                        GSPRINT4521_Convert(d_orig, (const unsigned char*)frame_recv->imagePtr, 
                                camera_params->width , camera_params->height, camera_params->width, camera_params->width, 0); // only for  8 bit 
                    
                        // cudaError_t cu_result = cudaMemcpy(d_orig, frame_recv->imagePtr, size_pic, cudaMemcpyDeviceToDevice);
                        // if (cu_result != cudaSuccess)
                        // {
                            // std::cout << "Cuda Error" << std::endl;
                        // }
                    }else{
                        EVT_FrameConvert(frame_recv, &FrameConvert, 0, 0, camera->linesReorderHandle);
                        // upload to gpu, can consider do this in a different thread, write encoder as a callback function?
                        cudaError_t cu_result = cudaMemcpy(d_orig, FrameConvert.imagePtr, size_pic, cudaMemcpyHostToDevice);
                        if (cu_result != cudaSuccess)
                        {
                            std::cout << "Cuda Error" << std::endl;
                        }
                    }
                }
                else{
                     if (camera_params->gpu_direct){
                        // upload to gpu, consider doing this in a different thread, write encoder as a callback function?
                        cudaError_t cu_result = cudaMemcpy(d_orig, frame_recv->imagePtr, size_pic, cudaMemcpyDeviceToDevice);
                        if (cu_result != cudaSuccess)
                        {
                            std::cout << "Cuda Error" << std::endl;
                        }
                    }else{
                        // upload to gpu, can consider do this in a different thread, write encoder as a callback function?
                        cudaError_t cu_result = cudaMemcpy(d_orig, frame_recv->imagePtr, size_pic, cudaMemcpyHostToDevice);
                        if (cu_result != cudaSuccess)
                        {
                            std::cout << "Cuda Error" << std::endl;
                        }
                    }
                }

                const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(d_orig,
                                                                     camera_params->width * sizeof(unsigned char),
                                                                     size,
                                                                     roi,
                                                                     d_debayer,
                                                                     camera_params->width * sizeof(uchar4),
                                                                     grid,
                                                                     NPPI_INTER_UNDEFINED,
                                                                     nAlpha);
                if (npp_result != 0)
                {
                    std::cout << "\nNPP error %d \n" << npp_result << std::endl;
                }

                // copy frame less 
                cudaError_t cu_result = cudaMemcpy2D(display_buffer, camera_params->width*4, d_debayer, camera_params->width*4, camera_params->width*4, camera_params->height, cudaMemcpyDeviceToDevice);
                if (cu_result != cudaSuccess)
                {
                    std::cout << "Cuda Error" << std::endl;
                }

                if(*encode_flag){
                    // encoding
                    const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
                    NvEncoderCuda::CopyToDeviceFrame(cuContext,
                                                    d_debayer,
                                                    0,
                                                    (CUdeviceptr)encoderInputFrame->inputPtr,
                                                    (int)encoderInputFrame->pitch,
                                                    pEnc->GetEncodeWidth(),
                                                    pEnc->GetEncodeHeight(),
                                                    CU_MEMORYTYPE_DEVICE,
                                                    encoderInputFrame->bufferFormat,
                                                    encoderInputFrame->chromaOffsets,
                                                    encoderInputFrame->numChromaPlanes);
                    pEnc->EncodeFrame(vPacket);
                    // num_frame_encode += (int)vPacket.size(); 
                    for (std::vector<uint8_t> &packet : vPacket)
                    {
                        // For each encoded packet
                        writer.Write(packet.data(), (int)packet.size(), num_frame_encode++);

                    }
                }

            }
        }
        else
        {
            dropped_frames++;
            std::cout << "EVT_CameraGetFrame Error" << camera_return << std::endl;
        }

        //In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if (frame_recv->frame_id == 65535)
            id_prev = 0;
        else
            id_prev = frame_recv->frame_id;

        if (camera_return)
            break; //No requeue reqd

        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.
        if (camera_return)
            std::cout << "EVT_CameraQueueFrame Error!" << std::endl;

        if (frame_count % 100 == 99)
        {
            printf(".");
            fflush(stdout);
        }
        if (frame_count % 10000 == 9999)
            printf("\n");

        if (dropped_frames >= 100)
            break;
        
        frame_count++; 
    }

    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));

    pEnc->EndEncode(vPacket);
    for (std::vector<uint8_t> &packet : vPacket)
    {
        writer.Write(packet.data(), (int)packet.size(), num_frame_encode++);
    }
    pEnc->DestroyEncoder();

    frame_metadata.close();
    double time_diff = w.Stop();

    //Report stats
    printf("\n");
    printf("Camera id: \t%d\n", camera_params->camera_id);
    printf("Frame count: \t%d\n", frame_count);
    printf("Frame received: \t%d\n", frames_recd);
    printf("Frame encoded: \t%d\n", num_frame_encode);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd / time_diff);

    // printf("Frame Rate Meas2: \t%f\n", ((float)(1000000000) * (float)(frame_count)) / ((float)(ptp_time_delta_sum)));
    // printf("Frame Rate Meas3: \t%f\n", ((float)(1000000000) * (float)(frame_count)) / ((float)(frame_ts_delta_sum)));
}
