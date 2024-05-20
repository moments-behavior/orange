#include "kernel.cuh"

/* Sensor process data:
    *Every group of 16 lines, change order to (0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15).
    *Then lines interleave:
    *0                        N
    *1                        0
    *2  top/bot interleave    N - 1
    *3  ==================>   1
    *                         ....
    *N
    *
**/
__global__ void GSPRINT4521_ConvertKernel(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift)
{
    //int x = threadIdx.x;  // 1
    //printf("threadIdx: x: %d y: %d\n", threadIdx.x, threadIdx.y);
    int row = blockIdx.x;  // 0 - (heigh - 1)
    const unsigned char* ptrS = src + row * strideS;

    bool isBot = !(row & 0x01); // line comes bot, top, bot, top
    int rowInHalf = row >> 1;  // top / bot interleave
    int rowBase16 = rowInHalf & 0xFFF0; // base row of group-16-line, 0, 16, 32, 48, ..., same for top and bottom

    int idGroup4 = (rowInHalf & 0xC) >> 2;  // 0 - 3 of group id in a group-16-line, start of each group-4-col
    int idInGroup4 = rowInHalf & 0x03;  // 0 - 3 in a group-4-col
    int lineInGroup16 = idGroup4 + (idInGroup4 << 2);  // idInGroup4 << 2 is idInGroup4 * 4

    int lineD = lineInGroup16 + rowBase16;

    if (isBot) {
        lineD = height - 1 - lineD;
    }
    unsigned char* ptrD = dest + lineD * strideD;

    if(leftShift == 0) { // not leftshift, just memcpy
        memcpy(ptrD, ptrS, strideS);
    } else {  // need left shift for 16 bit data
        unsigned short* ptrS16 = (unsigned short*)ptrS;
        unsigned short* ptrD16 = (unsigned short*)ptrD;
        for (int i = 0; i < width; i++) {
            unsigned short s = *ptrS16;
            *ptrD16 = (s << leftShift);
            ptrD16++;
            ptrS16++;
        }
    }
}

void GSPRINT4521_Convert(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift)
{
    dim3 threadPerBlock(1, 1);  // 1 thread per line    
//height = <blocksPerGrid>, 1 dimension, blockIdx.x == blockIdx.y == 0 ... (height - 1)
//threadPerBlock(1,2) = 2 dimensions, threadIdx.x == 0, threadIdx.y = 0 ... 1
    GSPRINT4521_ConvertKernel << <height, threadPerBlock >> > (dest, src, width, height, strideS, strideD, leftShift);  // this opens height blocks, 2 thread per block
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) printf("GSPRINT4521_ConvertKernel failed: %s\n", cudaGetErrorString(err));
}


__global__ void Mono8ToRGBMonoKernel(unsigned char* dest, const unsigned char* src, int width, int height)
{
    int y = blockIdx.x;  // y is the line
    int offsetMono = y * width;
    src += offsetMono;
    dest += offsetMono * 3;
    for (int i = 0; i < width; i++) {
        unsigned char c = src[i];
        *dest++ = c; *dest++ = c; *dest++ = c; // set the G to RGB
    }
}

void Mono8ToRGBMono(unsigned char* dest, const unsigned char* src, int width, int height)
{
    dim3 threadPerBlock(1, 1);  // 1 thread per line
    Mono8ToRGBMonoKernel << <height, threadPerBlock >> > (dest, src, width, height);  // this opens height blocks, 1 thread per block
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) printf("Mono8ToRGBMonoKernel failed: %s\n", cudaGetErrorString(err));
}


__global__ void rgba2rgb_kernel(unsigned char* dest, unsigned char* src, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x < width) && (y < height)) {
        *(dest + ((y * width * 3) + (x * 3))) = *(src + ((y * width * 4) + (x * 4)));            
        *(dest + ((y * width * 3) + (x * 3)) + 1) = *(src + ((y * width * 4) + (x * 4)) + 1);
        *(dest + ((y * width * 3) + (x * 3)) + 2) = *(src + ((y * width * 4) + (x * 4)) + 2);
    }
}



void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    rgba2rgb_kernel << <num_blocks, threads_per_block, 0, stream>> > (dest, src, width, height);
}


__global__ void rgba2bgr_kernel(unsigned char* dest, unsigned char* src, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x < width) && (y < height)) {
        *(dest + ((y * width * 3) + (x * 3))) = *(src + ((y * width * 4) + (x * 4)) + 2);            
        *(dest + ((y * width * 3) + (x * 3)) + 1) = *(src + ((y * width * 4) + (x * 4)) + 1);
        *(dest + ((y * width * 3) + (x * 3)) + 2) = *(src + ((y * width * 4) + (x * 4)));
    }
}

void rgba2bgr_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    rgba2bgr_kernel << <num_blocks, threads_per_block, 0, stream>> > (dest, src, width, height);
}


__global__ void gpu_draw_cicles(unsigned char* src, const int width, const int height, float* d_points, int num_points, const int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( (x < width) && (y < height) ) {
        for (int i= 0; i < num_points; i++) {
            if ((powf(x - d_points[i*2], 2) + powf(y - d_points[i*2+1], 2)) < (radius * radius))
            {
                *(src + ((y * width * 4) + (x * 4)))  = 255;
                *(src + ((y * width * 4) + (x * 4)) + 1)  = 0;
                *(src + ((y * width * 4) + (x * 4)) + 2)  = 255;
                *(src + ((y * width * 4) + (x * 4)) + 3)  = 255;   
            }
        }
    } 
}


void gpu_draw_cicles(unsigned char* src, int width, int height, float* d_points, int num_points, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    gpu_draw_cicles <<<num_blocks, threads_per_block, 0, stream>>> (src, width, height, d_points, num_points, 5);
}


__global__ void gpu_draw_box(unsigned char* src, const int width, const int height, float* d_points, double current_time)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if within image boundaries
    if( (x < width) && (y < height) ) {
        // loop over the four points of the bbox
        for (int i= 0; i < 4; i++) {

            float lengh_squared = (d_points[i*2+2]-d_points[i*2]) * (d_points[i*2+2]-d_points[i*2]) + (d_points[i*2+3]-d_points[i*2+1]) * (d_points[i*2+3]-d_points[i*2+1]);
            float dot_product = (x - d_points[i*2]) * (d_points[i*2+2] - d_points[i*2]) + (y-d_points[i*2+1]) * (d_points[i*2+3]-d_points[i*2+1]);
            float t = fmaxf(0.0f, fminf(1.0f, dot_product/lengh_squared));
            float proj_x = d_points[i*2] + t * (d_points[i*2+2] - d_points[i*2]);
            float proj_y = d_points[i*2+1] + t * (d_points[i*2+3]-d_points[i*2+1]);
            float distance_squared = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y);
            double multiplier = 0.5 * (sin(current_time * 0.00000001) + 1.0);
            if (distance_squared < 8.0f) {
                *(src + ((y * width * 4) + (x * 4)))  = 200 + (unsigned char) 55 * multiplier;
                *(src + ((y * width * 4) + (x * 4)) + 1)  = (unsigned char) 250 * multiplier;
                *(src + ((y * width * 4) + (x * 4)) + 2)  = (unsigned char) 255 * multiplier;
                *(src + ((y * width * 4) + (x * 4)) + 3)  = 255;                   
            }

        }
    } 
}


void gpu_draw_box(unsigned char* src, int width, int height, float* d_points, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    double current_time = (double) (std::chrono::system_clock::now().time_since_epoch()).count();
    gpu_draw_box <<<num_blocks, threads_per_block, 0, stream>>> (src, width, height, d_points, current_time);
}


__global__ void gpu_draw_rat_pose(unsigned char* src, const int width, const int height, float* d_points, unsigned int* d_skeleton)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;


    if( (x < width) && (y < height) ) {
        for (int i= 0; i < 4; i++) {
            unsigned int pt0_idx = d_skeleton[i*2];
            unsigned int pt1_idx = d_skeleton[i*2+1];
            float lengh_squared = (d_points[pt1_idx*2]-d_points[pt0_idx*2]) * (d_points[pt1_idx*2]-d_points[pt0_idx*2]) + (d_points[pt1_idx*2+1]-d_points[pt0_idx*2+1]) *  (d_points[pt1_idx*2+1]-d_points[pt0_idx*2+1]);
            float dot_product = (x - d_points[pt0_idx*2]) * (d_points[pt1_idx*2] - d_points[pt0_idx*2]) + (y-d_points[pt0_idx*2+1]) * (d_points[pt1_idx*2+1]-d_points[pt0_idx*2+1]);
            float t = fmaxf(0.0f, fminf(1.0f, dot_product/lengh_squared));
            float proj_x = d_points[pt0_idx*2] + t * (d_points[pt1_idx*2] - d_points[pt0_idx*2]);
            float proj_y = d_points[pt0_idx*2+1] + t * (d_points[pt1_idx*2+1]-d_points[pt0_idx*2+1]);
            float distance_squared = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y);
            if (distance_squared < 8.0f) {
                *(src + ((y * width * 4) + (x * 4)))  = 51;
                *(src + ((y * width * 4) + (x * 4)) + 1)  = 153;
                *(src + ((y * width * 4) + (x * 4)) + 2)  = 255;
                *(src + ((y * width * 4) + (x * 4)) + 3)  = 255;                   
            }
        }
    }
}


void gpu_draw_rat_pose(unsigned char* src, int width, int height, float* d_points, unsigned int* d_skeleton, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    gpu_draw_rat_pose <<<num_blocks, threads_per_block, 0, stream>>> (src, width, height, d_points, d_skeleton);
}


