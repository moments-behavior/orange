#include "video_capture_gpu.h"
#include "FFmpegWriter.h"



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
void aquire_frames_gpu_encode(Emergent::CEmergentCamera *camera, Emergent::CEmergentFrame *frame_recv, int num_frames, CameraParams camera_params, const char *output_file, const char *encoder_str, int gpu_index, CUdeviceptr dpFrame)
{
    int camera_return{0};

    unsigned int size_of_buffer;
    size_of_buffer = frame_recv->CalculateBufferSize();
    printf("Buffer size (bytes): \t%d\n ", size_of_buffer);

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;
    unsigned long long currentTimestamp, prevTimestamp, deltaTimestamp, prevDisplayTimestamp, deltaDisplayTimestamp;

    ck(cudaSetDevice(gpu_index));
    // modularize these parts: 1. debayer; 2. encoding; 
    // gpu: upload raw images and color debayer
    int output_channels = 4;
    unsigned char *d_orig;
    int size_pic = camera_params.width * camera_params.height * 1 * sizeof(unsigned char);
    cudaMalloc((void **)&d_orig, size_pic);
    unsigned char *d_debayer;
    cudaMalloc((void **)&d_debayer, size_pic * output_channels);

    // for encoding purpose 
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    NvEncoderInitParam encodeCLIOptions = NvEncoderInitParam(encoder_str);
    ck(cuInit(0));
    CUdevice cuDevice = 0;

    // specify which gpu
    ck(cuDeviceGet(&cuDevice, gpu_index));

    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
    //ck(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, camera_params.width, camera_params.height, eFormat));

    // debayer
    NppiSize size;
    size.width = camera_params.width;
    size.height = camera_params.height;
    Npp8u nAlpha = 255;

    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = camera_params.width;
    roi.height = camera_params.height;

    NppiBayerGridPosition grid;
    grid = NPPI_BAYER_RGGB;


    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);
    // For receiving encoded packets
    std::vector<std::vector<uint8_t>> vPacket;
    int num_frame_encode = 0;

    // for writing 
    FFmpegWriter writer(AV_CODEC_ID_H264, camera_params.width, camera_params.height, camera_params.frame_rate, output_file);
    
    // for displaying
    int nWidth = (camera_params.width + 1) & ~1; // make this a class, and attribute it 


    // start acquisition
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));
    StopWatch w;
    w.Start();
    for (int frame_count = 0; frame_count < num_frames; frame_count++)
    {
        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        if (!camera_return)
        {
            // timestamp 
            currentTimestamp = frame_recv->timestamp;
            printf("TimeStamp: %llu", currentTimestamp);


            deltaTimestamp =  currentTimestamp - prevTimestamp;
            prevTimestamp = currentTimestamp;

            //Counting dropped frames through frame_id as redundant check.
            if (((frame_recv->frame_id) != id_prev + 1) && (frame_count != 0))
                dropped_frames++;
            else
            {
                frames_recd++;
                // upload to gpu, can consider do this in a different thread, write encoder as a callback function?
                cudaError_t cu_result = cudaMemcpy(d_orig, frame_recv->imagePtr, size_pic, cudaMemcpyHostToDevice);
                if (cu_result != cudaSuccess)
                {
                    printf("Cuda Error");
                }

                const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(d_orig,
                                                                     camera_params.width * sizeof(unsigned char),
                                                                     size,
                                                                     roi,
                                                                     d_debayer,
                                                                     camera_params.width * sizeof(uchar4),
                                                                     grid,
                                                                     NPPI_INTER_UNDEFINED,
                                                                     nAlpha);
                if (npp_result != 0)
                {
                    printf("\nNPP error %d \n", npp_result);
                }


                // display at 60Hz 
                deltaDisplayTimestamp = currentTimestamp - prevDisplayTimestamp;

                if(deltaDisplayTimestamp > 6666666) // 60Hz monitor update
                {
                    prevDisplayTimestamp = prevTimestamp;
                    cudaMemcpy2D((uint8_t *)dpFrame, nWidth*4, d_debayer, nWidth*4, nWidth*4, camera_params.height, cudaMemcpyDeviceToDevice);
                }


                // copy to the dpFrame with display framerate 

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
        else
        {
            dropped_frames++;
            printf("\nEVT_CameraGetFrame Error = %8.8x!\n", camera_return);
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
            printf("EVT_CameraQueueFrame Error!\n");

        if (frame_count % 100 == 99)
        {
            printf(".");
            fflush(stdout);
        }
        if (frame_count % 10000 == 9999)
            printf("\n");

        if (dropped_frames >= 100)
            break;
    }

    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));
    pEnc->EndEncode(vPacket);
    for (std::vector<uint8_t> &packet : vPacket)
    {
        writer.Write(packet.data(), (int)packet.size(), num_frame_encode++);

    }
    pEnc->DestroyEncoder();
    double time_diff = w.Stop();
    //Report stats
    printf("\n");
    printf("Images Captured: \t%d\n", frames_recd);
    printf("Frame encoded: \t%d\n", num_frame_encode);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd / time_diff);
}
