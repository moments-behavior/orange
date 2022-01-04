#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>
#include "../Utils/NvCodecUtils.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderOutputInVidMemCuda.h"
#include "../Utils/Logger.h"
#include "../Utils/NvEncoderCLIOptions.h"

template<class EncoderClass> 
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
}


void EncodeCuda(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut)
{
    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, nWidth, nHeight, eFormat));
    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

    int nFrameSize = pEnc->GetFrameSize();
    printf("Frame size: %d \n", nFrameSize);

    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
    int nFrame = 0;
    while (true)
    {
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char*>(pHostFrame.get()), nFrameSize).gcount();
        // For receiving encoded packets
        std::vector<std::vector<uint8_t>> vPacket;
        if (nRead == nFrameSize)
        {
            const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();
            
            //struct timeval start, end;
            //gettimeofday(&start, NULL);
            //auto start = high_resolution_clock::now();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                (int)encoderInputFrame->pitch,
                pEnc->GetEncodeWidth(),
                pEnc->GetEncodeHeight(),
                CU_MEMORYTYPE_HOST, 
                encoderInputFrame->bufferFormat,
                encoderInputFrame->chromaOffsets,
                encoderInputFrame->numChromaPlanes);
            pEnc->EncodeFrame(vPacket);
        }
        else
        {
            pEnc->EndEncode(vPacket);
        }
        nFrame += (int)vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket)
        {
            // For each encoded packet
            fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
        }

        if (nRead != nFrameSize) break;
    }

    pEnc->DestroyEncoder();

    std::cout << "Total frames encoded: " << nFrame << std::endl;
}