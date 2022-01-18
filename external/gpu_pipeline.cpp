
#define STB_IMAGE_IMPLEMENTATION
#include "NvEncoder/stb_image.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nppi.h>
#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegWriter.h"

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

int main()
{
    char szOutFilePath[256] = "frames_encode.mp4"; // the header file is a bit wacky
    int width = 3208;
    int height = 2200;
    int channel = 1;
    std::string frame_name{};

    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    int iGpu = 0;
    NvEncoderInitParam encodeCLIOptions = NvEncoderInitParam("-preset p1 -fps 50");
    ck(cuInit(0));
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    // Open output file
    // std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    // if (!fpOut)
    // {
    //     std::ostringstream err;
    //     err << "Unable to open output file: " << szOutFilePath << std::endl;
    //     throw std::invalid_argument(err.str());
    // }

    // try ffmpeg writer 
    const char *szMediaUri = "frames_encode.mp4";
    FFmpegWriter writer(AV_CODEC_ID_H264, width, height, 30, szMediaUri);
    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, width, height, eFormat));

    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);
    // For receiving encoded packets
    std::vector<std::vector<uint8_t>> vPacket;
    
    unsigned char *d_orig;
    int size_pic = width * height * channel * sizeof(unsigned char);
    cudaMalloc((void **)&d_orig, size_pic);
    
    int output_channels = 4;
    unsigned char *d_debayer;
    cudaMalloc((void **)&d_debayer, size_pic * output_channels);
    // debayer
    NppiSize size;
    size.width = width;
    size.height = height;

    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = width;
    roi.height = height;

    NppiBayerGridPosition grid;
    grid = NPPI_BAYER_RGGB;
    Npp8u nAlpha = 255;

    // allocate memory for picture array 
    std::vector<unsigned char*> frame_vector (100);

    std::cout << "Loading" << std::endl;
    for (int picture_count = 0; picture_count < 100; picture_count++)
    {
        // load pictures first
        frame_name = "/home/ash/src/orange/frames/frame_" + std::to_string(picture_count+1) + ".bmp";
        frame_vector[picture_count] = stbi_load(frame_name.c_str(), &width, &height, &channel, 1);
    }

    //int num_frame_encode = 0;
    int nFrame = 0;
    StopWatch w;
    w.Start();
    std::cout << "Start encoding..." << std::endl;
    for (int frame_id = 0; frame_id < 100; frame_id++)
    {
        channel = 1;
        // load next frames
        cudaError_t result = cudaMemcpy(d_orig, frame_vector[frame_id], size_pic, cudaMemcpyHostToDevice);

        //std::cout << frame_id << std::endl;
        if (result != cudaSuccess)
        {
            std::cout << "Cuda Error, here?" << frame_id << std::endl;
        }

        const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(d_orig,
                                                             width * sizeof(unsigned char),
                                                             size,
                                                             roi,
                                                             d_debayer,
                                                             width * sizeof(uchar4),
                                                             grid,
                                                             NPPI_INTER_UNDEFINED,
                                                             nAlpha);
        if (npp_result != 0)
        {
            printf("\nNPP error %d \n", npp_result);
        }
        // cudaDeviceSynchronize();
        // encode
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
        //num_frame_encode += (int)vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket)
        {
            // For each encoded packet
            //fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
            writer.Write(packet.data(), (int)packet.size(), nFrame++);

        }
    }

    pEnc->EndEncode(vPacket);
    //num_frame_encode += (int)vPacket.size();
    for (std::vector<uint8_t> &packet : vPacket)
    {
        // For each encoded packet
        //fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
        writer.Write(packet.data(), (int)packet.size(), nFrame++);

    }
    pEnc->DestroyEncoder();
    //fpOut.close();
    double t = w.Stop();
    std::cout << "Total frames encoded: " << nFrame << " frames, in: " << t << " seconds, " << "FPS: " << (double)nFrame/t << std::endl;
    std::cout << "Bitstream saved in file " << szOutFilePath << std::endl;
}