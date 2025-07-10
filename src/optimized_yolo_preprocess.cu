// optimized_yolo_preprocess.cu
// Single kernel to replace the entire preprocessing pipeline
// Mono 4512x4512 -> BGR 640x640 -> Normalized Float Planar in one pass

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Optimized kernel that does everything in one pass:
// 1. Read from large mono image (4512x4512)
// 2. Resize with bilinear interpolation to target size
// 3. Convert mono to BGR
// 4. Add letterbox padding
// 5. Normalize to [0,1] 
// 6. Convert to planar format (CCCCC instead of CRGBCRGB)
__global__ void mono_to_yolo_optimized(
    const unsigned char* __restrict__ src_mono,    // Input: 4512x4512 mono
    float* __restrict__ dst_planar,                // Output: 640x640x3 planar float
    int src_width,                                 // 4512
    int src_height,                                // 4512  
    int dst_width,                                 // 640
    int dst_height,                                // 640
    float scale_x,                                 // scaling factors
    float scale_y,
    int pad_left,                                  // letterbox padding
    int pad_top)
{
    // Calculate output pixel coordinates
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    float pixel_value = 114.0f; // Default padding value
    
    // Check if we're inside the actual image area (not padding)
    if (dst_x >= pad_left && dst_x < (dst_width - pad_left) && 
        dst_y >= pad_top && dst_y < (dst_height - pad_top)) {
        
        // Map back to source coordinates with bilinear interpolation
        float src_x_f = (dst_x - pad_left) * scale_x;
        float src_y_f = (dst_y - pad_top) * scale_y;
        
        int src_x0 = (int)floorf(src_x_f);
        int src_y0 = (int)floorf(src_y_f);
        int src_x1 = min(src_x0 + 1, src_width - 1);
        int src_y1 = min(src_y0 + 1, src_height - 1);
        
        float wx = src_x_f - src_x0;
        float wy = src_y_f - src_y0;
        
        // Bounds checking
        src_x0 = max(0, min(src_x0, src_width - 1));
        src_y0 = max(0, min(src_y0, src_height - 1));
        
        // Bilinear interpolation
        unsigned char p00 = src_mono[src_y0 * src_width + src_x0];
        unsigned char p01 = src_mono[src_y0 * src_width + src_x1];
        unsigned char p10 = src_mono[src_y1 * src_width + src_x0];
        unsigned char p11 = src_mono[src_y1 * src_width + src_x1];
        
        float p0 = p00 * (1.0f - wx) + p01 * wx;
        float p1 = p10 * (1.0f - wx) + p11 * wx;
        pixel_value = p0 * (1.0f - wy) + p1 * wy;
    }
    
    // Normalize to [0,1] range
    pixel_value /= 255.0f;
    
    // Calculate output indices for planar format
    int pixel_idx = dst_y * dst_width + dst_x;
    int channel_size = dst_width * dst_height;
    
    // For mono->BGR: R=G=B=mono_value
    dst_planar[pixel_idx] = pixel_value;                    // B channel
    dst_planar[pixel_idx + channel_size] = pixel_value;     // G channel  
    dst_planar[pixel_idx + 2 * channel_size] = pixel_value; // R channel
}

// Alternative version for color images (if needed later)
__global__ void bayer_to_yolo_optimized(
    const unsigned char* __restrict__ src_bayer,
    float* __restrict__ dst_planar,
    int src_width, int src_height,
    int dst_width, int dst_height,
    float scale_x, float scale_y,
    int pad_left, int pad_top,
    int bayer_pattern) // 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    float r_val = 114.0f, g_val = 114.0f, b_val = 114.0f;
    
    if (dst_x >= pad_left && dst_x < (dst_width - pad_left) && 
        dst_y >= pad_top && dst_y < (dst_height - pad_top)) {
        
        float src_x_f = (dst_x - pad_left) * scale_x;
        float src_y_f = (dst_y - pad_top) * scale_y;
        
        int src_x = (int)roundf(src_x_f);
        int src_y = (int)roundf(src_y_f);
        
        src_x = max(0, min(src_x, src_width - 1));
        src_y = max(0, min(src_y, src_height - 1));
        
        // Simple demosaicing (nearest neighbor)
        unsigned char pixel = src_bayer[src_y * src_width + src_x];
        
        // Determine color based on position and bayer pattern
        int color_idx = ((src_y & 1) << 1) | (src_x & 1);
        if (bayer_pattern == 1) color_idx ^= 1; // GRBG
        if (bayer_pattern == 2) color_idx ^= 2; // GBRG  
        if (bayer_pattern == 3) color_idx ^= 3; // BGGR
        
        switch(color_idx) {
            case 0: r_val = pixel; g_val = pixel * 0.8f; b_val = pixel * 0.8f; break; // R
            case 1: case 2: r_val = pixel * 0.9f; g_val = pixel; b_val = pixel * 0.9f; break; // G
            case 3: r_val = pixel * 0.8f; g_val = pixel * 0.8f; b_val = pixel; break; // B
        }
    }
    
    // Normalize and store in planar format
    int pixel_idx = dst_y * dst_width + dst_x;
    int channel_size = dst_width * dst_height;
    
    dst_planar[pixel_idx] = b_val / 255.0f;                    // B
    dst_planar[pixel_idx + channel_size] = g_val / 255.0f;     // G
    dst_planar[pixel_idx + 2 * channel_size] = r_val / 255.0f; // R
}

// Host wrapper function
extern "C" {
    void launch_optimized_yolo_preprocess(
        const unsigned char* d_src,
        float* d_dst_planar,
        int src_width, int src_height,
        int dst_width, int dst_height,
        bool is_color,
        cudaStream_t stream)
    {
        // Calculate scaling and padding for letterbox
        float scale = fminf((float)dst_width / src_width, (float)dst_height / src_height);
        int scaled_width = (int)(src_width * scale);
        int scaled_height = (int)(src_height * scale);
        int pad_left = (dst_width - scaled_width) / 2;
        int pad_top = (dst_height - scaled_height) / 2;
        
        float scale_x = (float)src_width / scaled_width;
        float scale_y = (float)src_height / scaled_height;
        
        // Launch kernel
        dim3 block(16, 16);
        dim3 grid((dst_width + block.x - 1) / block.x, 
                  (dst_height + block.y - 1) / block.y);
        
        if (is_color) {
            bayer_to_yolo_optimized<<<grid, block, 0, stream>>>(
                d_src, d_dst_planar, src_width, src_height,
                dst_width, dst_height, scale_x, scale_y,
                pad_left, pad_top, 0); // Assuming RGGB for now
        } else {
            mono_to_yolo_optimized<<<grid, block, 0, stream>>>(
                d_src, d_dst_planar, src_width, src_height,
                dst_width, dst_height, scale_x, scale_y,
                pad_left, pad_top);
        }
    }
}

// Helper function to calculate memory requirements
extern "C" {
    size_t get_optimized_output_size(int dst_width, int dst_height) {
        return dst_width * dst_height * 3 * sizeof(float);
    }
}