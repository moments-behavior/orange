#include "jarvis_pose_det.h"
#include "realtime_tool.h"
#include "common.hpp"
#include "utils.h"
#include <npp.h>
#include <nvToolsExt.h>
#include <fstream>
#include <iostream>

JarvisPoseDetector::JarvisPoseDetector(const std::string &model_dir,
                                       int num_cameras,
                                       cudaStream_t stream)
    : num_cameras(num_cameras), stream(stream) {
    
    // Initialize default values
    center_img_size = JARVIS_CENTER_IMG_SIZE;
    keypoint_img_size = JARVIS_KEYPOINT_IMG_SIZE;
    num_keypoints = JARVIS_NUM_KEYPOINTS;
    
    // Default ImageNet normalization
    mean_values = {0.485f, 0.456f, 0.406f};
    std_values = {0.229f, 0.224f, 0.225f};
    
    // Initialize results storage
    center_results.resize(num_cameras);
    center_confidences.resize(num_cameras);
    keypoint_results.resize(num_cameras * num_keypoints);
    keypoint_confidences.resize(num_cameras * num_keypoints);
    pose_3d_results.resize(num_keypoints);
    pose_3d_confidences.resize(num_keypoints);
    
    // Load model information
    load_model_info(model_dir);
    
    // Load TensorRT engines
    load_models(model_dir);
}

JarvisPoseDetector::~JarvisPoseDetector() {
    // Clean up TensorRT resources
    if (center_context) {
        delete center_context;
        center_context = nullptr;
    }
    if (keypoint_context) {
        delete keypoint_context;
        keypoint_context = nullptr;
    }
    if (hybrid_net_context) {
        delete hybrid_net_context;
        hybrid_net_context = nullptr;
    }
    
    if (center_engine) {
        delete center_engine;
        center_engine = nullptr;
    }
    if (keypoint_engine) {
        delete keypoint_engine;
        keypoint_engine = nullptr;
    }
    if (hybrid_net_engine) {
        delete hybrid_net_engine;
        hybrid_net_engine = nullptr;
    }
    
    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }
    
    // Clean up device memory
    for (auto &ptr : center_device_ptrs) {
        if (ptr) CHECK(cudaFreeAsync(ptr, stream));
    }
    for (auto &ptr : center_host_ptrs) {
        if (ptr) CHECK(cudaFreeHost(ptr));
    }
    for (auto &ptr : keypoint_device_ptrs) {
        if (ptr) CHECK(cudaFreeAsync(ptr, stream));
    }
    for (auto &ptr : keypoint_host_ptrs) {
        if (ptr) CHECK(cudaFreeHost(ptr));
    }
    for (auto &ptr : hybrid_net_device_ptrs) {
        if (ptr) CHECK(cudaFreeAsync(ptr, stream));
    }
    for (auto &ptr : hybrid_net_host_ptrs) {
        if (ptr) CHECK(cudaFreeHost(ptr));
    }
    
    // Clean up preprocessing buffers
    if (d_resized_images) CHECK(cudaFreeAsync(d_resized_images, stream));
    if (d_cropped_images) CHECK(cudaFreeAsync(d_cropped_images, stream));
    if (d_float_images) CHECK(cudaFreeAsync(d_float_images, stream));
}

void JarvisPoseDetector::load_model_info(const std::string &model_dir) {
    // For now, use default values since JSON library might not be available
    // In a full implementation, this would load from model_info.json
    std::cout << "Using default Jarvis model configuration: " << num_cameras << " cameras, " 
              << num_keypoints << " keypoints" << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
}

void JarvisPoseDetector::load_models(const std::string &model_dir) {
    // Initialize TensorRT runtime
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    
    // Load center detection model
    std::string center_path = model_dir + "/center_detect.engine";
    std::ifstream center_file(center_path, std::ios::binary);
    if (center_file.good()) {
        center_file.seekg(0, std::ios::end);
        size_t center_size = center_file.tellg();
        center_file.seekg(0, std::ios::beg);
        
        std::vector<char> center_data(center_size);
        center_file.read(center_data.data(), center_size);
        
        center_engine = runtime->deserializeCudaEngine(center_data.data(), center_size);
        if (center_engine) {
            center_context = center_engine->createExecutionContext();
            std::cout << "Loaded center detection model" << std::endl;
        }
    }
    
    // Load keypoint detection model
    std::string keypoint_path = model_dir + "/keypoint_detect.engine";
    std::ifstream keypoint_file(keypoint_path, std::ios::binary);
    if (keypoint_file.good()) {
        keypoint_file.seekg(0, std::ios::end);
        size_t keypoint_size = keypoint_file.tellg();
        keypoint_file.seekg(0, std::ios::beg);
        
        std::vector<char> keypoint_data(keypoint_size);
        keypoint_file.read(keypoint_data.data(), keypoint_size);
        
        keypoint_engine = runtime->deserializeCudaEngine(keypoint_data.data(), keypoint_size);
        if (keypoint_engine) {
            keypoint_context = keypoint_engine->createExecutionContext();
            std::cout << "Loaded keypoint detection model" << std::endl;
        }
    }
    
    // Load hybrid net model
    std::string hybrid_path = model_dir + "/hybrid_net.engine";
    std::ifstream hybrid_file(hybrid_path, std::ios::binary);
    if (hybrid_file.good()) {
        hybrid_file.seekg(0, std::ios::end);
        size_t hybrid_size = hybrid_file.tellg();
        hybrid_file.seekg(0, std::ios::beg);
        
        std::vector<char> hybrid_data(hybrid_size);
        hybrid_file.read(hybrid_data.data(), hybrid_size);
        
        hybrid_net_engine = runtime->deserializeCudaEngine(hybrid_data.data(), hybrid_size);
        if (hybrid_net_engine) {
            hybrid_net_context = hybrid_net_engine->createExecutionContext();
            std::cout << "Loaded hybrid net model" << std::endl;
        }
    }
    
    // Allocate memory for all models
    make_pipe(false);
}

void JarvisPoseDetector::make_pipe(bool graph_capture) {
    // Allocate memory for center detection
    if (center_engine) {
        int num_bindings = center_engine->getNbIOTensors();
        for (int i = 0; i < num_bindings; ++i) {
            void *d_ptr, *h_ptr;
            auto shape = center_engine->getTensorShape(center_engine->getIOTensorName(i));
            size_t size = 1;
            for (int j = 0; j < shape.nbDims; j++) {
                size *= shape.d[j];
            }
            size *= center_engine->getTensorDataType(center_engine->getIOTensorName(i)) == nvinfer1::DataType::kFLOAT ? 4 : 1;
            
            CHECK(cudaMallocAsync(&d_ptr, size, stream));
            CHECK(cudaHostAlloc(&h_ptr, size, 0));
            
            center_device_ptrs.push_back(d_ptr);
            center_host_ptrs.push_back(h_ptr);
        }
    }
    
    // Allocate memory for keypoint detection
    if (keypoint_engine) {
        int num_bindings = keypoint_engine->getNbIOTensors();
        for (int i = 0; i < num_bindings; ++i) {
            void *d_ptr, *h_ptr;
            auto shape = keypoint_engine->getTensorShape(keypoint_engine->getIOTensorName(i));
            size_t size = 1;
            for (int j = 0; j < shape.nbDims; j++) {
                size *= shape.d[j];
            }
            size *= keypoint_engine->getTensorDataType(keypoint_engine->getIOTensorName(i)) == nvinfer1::DataType::kFLOAT ? 4 : 1;
            
            CHECK(cudaMallocAsync(&d_ptr, size, stream));
            CHECK(cudaHostAlloc(&h_ptr, size, 0));
            
            keypoint_device_ptrs.push_back(d_ptr);
            keypoint_host_ptrs.push_back(h_ptr);
        }
    }
    
    // Allocate memory for hybrid net
    if (hybrid_net_engine) {
        int num_bindings = hybrid_net_engine->getNbIOTensors();
        for (int i = 0; i < num_bindings; ++i) {
            void *d_ptr, *h_ptr;
            auto shape = hybrid_net_engine->getTensorShape(hybrid_net_engine->getIOTensorName(i));
            size_t size = 1;
            for (int j = 0; j < shape.nbDims; j++) {
                size *= shape.d[j];
            }
            size *= hybrid_net_engine->getTensorDataType(hybrid_net_engine->getIOTensorName(i)) == nvinfer1::DataType::kFLOAT ? 4 : 1;
            
            CHECK(cudaMallocAsync(&d_ptr, size, stream));
            CHECK(cudaHostAlloc(&h_ptr, size, 0));
            
            hybrid_net_device_ptrs.push_back(d_ptr);
            hybrid_net_host_ptrs.push_back(h_ptr);
        }
    }
    
    // Allocate preprocessing buffers
    size_t resized_size = num_cameras * 3 * center_img_size * center_img_size * sizeof(unsigned char);
    size_t cropped_size = num_cameras * 3 * keypoint_img_size * keypoint_img_size * sizeof(unsigned char);
    size_t float_size = num_cameras * 3 * center_img_size * center_img_size * sizeof(float);
    
    CHECK(cudaMallocAsync(&d_resized_images, resized_size, stream));
    CHECK(cudaMallocAsync(&d_cropped_images, cropped_size, stream));
    CHECK(cudaMallocAsync(&d_float_images, float_size, stream));
}

void JarvisPoseDetector::detect_centers(unsigned char **d_input_images, int img_width, int img_height) {
    if (!center_engine || !center_context) {
        std::cerr << "Center detection model not loaded" << std::endl;
        return;
    }
    
    // Preprocess images for center detection
    preprocess_center_detection(d_input_images, img_width, img_height);
    
    // Run inference
    for (int32_t i = 0, e = center_engine->getNbIOTensors(); i < e; i++) {
        auto const name = center_engine->getIOTensorName(i);
        center_context->setTensorAddress(name, center_device_ptrs[i]);
    }
    
    center_context->enqueueV3(stream);
    
    // Copy results to host
    for (int i = 0; i < center_engine->getNbIOTensors() - center_engine->getNbIOTensors()/2; i++) {
        auto shape = center_engine->getTensorShape(center_engine->getIOTensorName(i + center_engine->getNbIOTensors()/2));
        size_t osize = 1;
        for (int j = 0; j < shape.nbDims; j++) {
            osize *= shape.d[j];
        }
        osize *= 4;
        CHECK(cudaMemcpyAsync(center_host_ptrs[i], center_device_ptrs[i + center_engine->getNbIOTensors()/2], 
                              osize, cudaMemcpyDeviceToHost, stream));
    }
    
    cudaStreamSynchronize(stream);
    
    // Postprocess results
    postprocess_center_detection();
}

void JarvisPoseDetector::detect_keypoints(unsigned char **d_input_images, int img_width, int img_height) {
    if (!keypoint_engine || !keypoint_context) {
        std::cerr << "Keypoint detection model not loaded" << std::endl;
        return;
    }
    
    // Preprocess images for keypoint detection
    preprocess_keypoint_detection(d_input_images, img_width, img_height);
    
    // Run inference
    for (int32_t i = 0, e = keypoint_engine->getNbIOTensors(); i < e; i++) {
        auto const name = keypoint_engine->getIOTensorName(i);
        keypoint_context->setTensorAddress(name, keypoint_device_ptrs[i]);
    }
    
    keypoint_context->enqueueV3(stream);
    
    // Copy results to host
    for (int i = 0; i < keypoint_engine->getNbIOTensors() - keypoint_engine->getNbIOTensors()/2; i++) {
        auto shape = keypoint_engine->getTensorShape(keypoint_engine->getIOTensorName(i + keypoint_engine->getNbIOTensors()/2));
        size_t osize = 1;
        for (int j = 0; j < shape.nbDims; j++) {
            osize *= shape.d[j];
        }
        osize *= 4;
        CHECK(cudaMemcpyAsync(keypoint_host_ptrs[i], keypoint_device_ptrs[i + keypoint_engine->getNbIOTensors()/2], 
                              osize, cudaMemcpyDeviceToHost, stream));
    }
    
    cudaStreamSynchronize(stream);
    
    // Postprocess results
    postprocess_keypoint_detection();
}

void JarvisPoseDetector::reconstruct_3d_pose() {
    if (!hybrid_net_engine || !hybrid_net_context) {
        std::cerr << "Hybrid net model not loaded" << std::endl;
        return;
    }
    
    // Triangulate 3D center first
    triangulate_3d_center();
    
    // Run hybrid net inference for 3D reconstruction
    for (int32_t i = 0, e = hybrid_net_engine->getNbIOTensors(); i < e; i++) {
        auto const name = hybrid_net_engine->getIOTensorName(i);
        hybrid_net_context->setTensorAddress(name, hybrid_net_device_ptrs[i]);
    }
    
    hybrid_net_context->enqueueV3(stream);
    
    // Copy results to host
    for (int i = 0; i < hybrid_net_engine->getNbIOTensors() - hybrid_net_engine->getNbIOTensors()/2; i++) {
        auto shape = hybrid_net_engine->getTensorShape(hybrid_net_engine->getIOTensorName(i + hybrid_net_engine->getNbIOTensors()/2));
        size_t osize = 1;
        for (int j = 0; j < shape.nbDims; j++) {
            osize *= shape.d[j];
        }
        osize *= 4;
        CHECK(cudaMemcpyAsync(hybrid_net_host_ptrs[i], hybrid_net_device_ptrs[i + hybrid_net_engine->getNbIOTensors()/2], 
                              osize, cudaMemcpyDeviceToHost, stream));
    }
    
    cudaStreamSynchronize(stream);
    
    // Postprocess 3D reconstruction
    postprocess_3d_reconstruction();
}

void JarvisPoseDetector::preprocess_center_detection(unsigned char **d_input_images, int img_width, int img_height) {
    // Resize images to center detection size
    NppiSize src_size = {img_width, img_height};
    NppiSize dst_size = {center_img_size, center_img_size};
    NppiRect src_roi = {0, 0, img_width, img_height};
    NppiRect dst_roi = {0, 0, center_img_size, center_img_size};
    
    for (int i = 0; i < num_cameras; ++i) {
        NppStatus status = nppiResize_8u_C3R_Ctx(
            d_input_images[i], img_width * sizeof(uchar3), src_size, src_roi,
            d_resized_images + i * 3 * center_img_size * center_img_size,
            center_img_size * sizeof(uchar3), dst_size, dst_roi,
            NPPI_INTER_LINEAR, {stream, 0}
        );
        
        if (status != NPP_SUCCESS) {
            std::cerr << "Error resizing image for camera " << i << std::endl;
        }
    }
    
    // Convert to float and normalize
    Npp32f scale[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    Npp32f shift[3] = {-mean_values[0], -mean_values[1], -mean_values[2]};
    
    for (int i = 0; i < num_cameras; ++i) {
        NppStatus status = nppiConvert_8u32f_C3R_Ctx(
            d_resized_images + i * 3 * center_img_size * center_img_size,
            center_img_size * sizeof(uchar3),
            d_float_images + i * 3 * center_img_size * center_img_size,
            center_img_size * sizeof(float3),
            dst_size, {stream, 0}
        );
        
        if (status == NPP_SUCCESS) {
            status = nppiMulC_32f_C3IR_Ctx(scale, d_float_images + i * 3 * center_img_size * center_img_size,
                                          center_img_size * sizeof(float3), dst_size, {stream, 0});
            if (status == NPP_SUCCESS) {
                status = nppiAddC_32f_C3IR_Ctx(shift, d_float_images + i * 3 * center_img_size * center_img_size,
                                              center_img_size * sizeof(float3), dst_size, {stream, 0});
            }
        }
        
        if (status != NPP_SUCCESS) {
            std::cerr << "Error preprocessing image for camera " << i << std::endl;
        }
    }
}

void JarvisPoseDetector::preprocess_keypoint_detection(unsigned char **d_input_images, int img_width, int img_height) {
    // Crop regions around detected centers and preprocess for keypoint detection
    // Based on the Jarvis pipeline: crop 192x192 regions around detected centers
    
    if (center_results.empty()) {
        std::cout << "No center detections available for keypoint cropping" << std::endl;
        return;
    }
    
    // For each camera, crop region around detected center
    for (int i = 0; i < num_cameras && i < center_results.size(); i++) {
        if (center_results[i].x < 0 || center_results[i].y < 0) {
            continue; // Skip invalid detections
        }
        
        // Calculate crop region around center
        int crop_size = keypoint_img_size; // 192x192
        int half_crop = crop_size / 2;
        
        int center_x = static_cast<int>(center_results[i].x);
        int center_y = static_cast<int>(center_results[i].y);
        
        // Ensure crop region is within image bounds
        int crop_x = std::max(0, std::min(img_width - crop_size, center_x - half_crop));
        int crop_y = std::max(0, std::min(img_height - crop_size, center_y - half_crop));
        
        // Crop and resize to keypoint detection size
        NppiSize src_size = {crop_size, crop_size};
        NppiSize dst_size = {keypoint_img_size, keypoint_img_size};
        NppiRect src_roi = {0, 0, crop_size, crop_size};
        NppiRect dst_roi = {0, 0, keypoint_img_size, keypoint_img_size};
        
        // Note: This is a simplified implementation
        // In practice, you'd need to implement proper GPU cropping
        // For now, we'll use the full image as a placeholder
        NppStatus status = nppiResize_8u_C3R_Ctx(
            d_input_images[i], img_width * sizeof(uchar3), src_size, src_roi,
            d_cropped_images + i * 3 * keypoint_img_size * keypoint_img_size,
            keypoint_img_size * sizeof(uchar3), dst_size, dst_roi,
            NPPI_INTER_LINEAR, {stream, 0}
        );
        
        if (status != NPP_SUCCESS) {
            std::cerr << "Error cropping image for camera " << i << std::endl;
        }
    }
    
    // Convert cropped images to float and normalize (similar to center detection)
    Npp32f scale[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    Npp32f shift[3] = {-mean_values[0], -mean_values[1], -mean_values[2]};
    
    for (int i = 0; i < num_cameras; i++) {
        NppStatus status = nppiConvert_8u32f_C3R_Ctx(
            d_cropped_images + i * 3 * keypoint_img_size * keypoint_img_size,
            keypoint_img_size * sizeof(uchar3),
            d_float_images + i * 3 * keypoint_img_size * keypoint_img_size,
            keypoint_img_size * sizeof(float3),
            dst_size, {stream, 0}
        );
        
        if (status == NPP_SUCCESS) {
            status = nppiMulC_32f_C3IR_Ctx(scale, d_float_images + i * 3 * keypoint_img_size * keypoint_img_size,
                                          keypoint_img_size * sizeof(float3), dst_size, {stream, 0});
            if (status == NPP_SUCCESS) {
                status = nppiAddC_32f_C3IR_Ctx(shift, d_float_images + i * 3 * keypoint_img_size * keypoint_img_size,
                                              keypoint_img_size * sizeof(float3), dst_size, {stream, 0});
            }
        }
        
        if (status != NPP_SUCCESS) {
            std::cerr << "Error preprocessing keypoint image for camera " << i << std::endl;
        }
    }
}

void JarvisPoseDetector::postprocess_center_detection() {
    // Parse center detection heatmap outputs
    // The center detection model outputs a heatmap of size [num_cameras, 1, H, W]
    // We need to find the peak in each camera's heatmap
    
    center_results.clear();
    center_confidences.clear();
    
    if (!center_engine || center_host_ptrs.empty()) {
        return;
    }
    
    // Get output tensor shape
    int output_idx = center_engine->getNbIOTensors() / 2; // First half are inputs
    auto output_shape = center_engine->getTensorShape(center_engine->getIOTensorName(output_idx));
    
    // Expected shape: [num_cameras, 1, height, width]
    int batch_size = output_shape.d[0];
    int channels = output_shape.d[1];
    int height = output_shape.d[2];
    int width = output_shape.d[3];
    
    float* output_data = static_cast<float*>(center_host_ptrs[0]);
    
    for (int cam = 0; cam < batch_size; cam++) {
        // Find peak in heatmap for this camera
        float max_val = 0.0f;
        int max_x = 0, max_y = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = cam * channels * height * width + y * width + x;
                float val = output_data[idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_x = x;
                    max_y = y;
                }
            }
        }
        
        // Convert heatmap coordinates to image coordinates
        // Scale from heatmap size to original image size
        float scale_x = static_cast<float>(center_img_size) / width;
        float scale_y = static_cast<float>(center_img_size) / height;
        
        cv::Point2f center(max_x * scale_x, max_y * scale_y);
        float confidence = max_val;
        
        // Apply confidence threshold
        if (confidence > 0.5f) {
            center_results.push_back(center);
            center_confidences.push_back(confidence);
        } else {
            center_results.push_back(cv::Point2f(-1, -1)); // Invalid detection
            center_confidences.push_back(0.0f);
        }
    }
}

void JarvisPoseDetector::postprocess_keypoint_detection() {
    // Parse keypoint detection heatmap outputs
    // The keypoint detection model outputs heatmaps of size [num_cameras, num_keypoints, H, W]
    // We need to find the peak for each keypoint in each camera's heatmap
    
    keypoint_results.clear();
    keypoint_confidences.clear();
    
    if (!keypoint_engine || keypoint_host_ptrs.empty()) {
        return;
    }
    
    // Get output tensor shape
    int output_idx = keypoint_engine->getNbIOTensors() / 2; // First half are inputs
    auto output_shape = keypoint_engine->getTensorShape(keypoint_engine->getIOTensorName(output_idx));
    
    // Expected shape: [num_cameras, num_keypoints, height, width]
    int batch_size = output_shape.d[0];
    int num_keypoints = output_shape.d[1];
    int height = output_shape.d[2];
    int width = output_shape.d[3];
    
    float* output_data = static_cast<float*>(keypoint_host_ptrs[0]);
    
    for (int cam = 0; cam < batch_size; cam++) {
        for (int kp = 0; kp < num_keypoints; kp++) {
            // Find peak in heatmap for this keypoint
            float max_val = 0.0f;
            int max_x = 0, max_y = 0;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = cam * num_keypoints * height * width + 
                              kp * height * width + y * width + x;
                    float val = output_data[idx];
                    
                    if (val > max_val) {
                        max_val = val;
                        max_x = x;
                        max_y = y;
                    }
                }
            }
            
            // Convert heatmap coordinates to image coordinates
            // Scale from heatmap size to cropped region size
            float scale_x = static_cast<float>(keypoint_img_size) / width;
            float scale_y = static_cast<float>(keypoint_img_size) / height;
            
            cv::Point2f keypoint(max_x * scale_x, max_y * scale_y);
            float confidence = max_val;
            
            // Apply confidence threshold
            if (confidence > 0.5f) {
                keypoint_results.push_back(keypoint);
                keypoint_confidences.push_back(confidence);
            } else {
                keypoint_results.push_back(cv::Point2f(-1, -1)); // Invalid detection
                keypoint_confidences.push_back(0.0f);
            }
        }
    }
}

void JarvisPoseDetector::postprocess_3d_reconstruction() {
    // Parse 3D reconstruction outputs from HybridNet
    // The HybridNet outputs 3D keypoint coordinates and confidences
    // Based on the CSV output format: [x, y, z, confidence] for each keypoint
    
    pose_3d_results.clear();
    pose_3d_confidences.clear();
    
    if (!hybrid_net_engine || hybrid_net_host_ptrs.empty()) {
        return;
    }
    
    // Get output tensor shape
    int output_idx = hybrid_net_engine->getNbIOTensors() / 2; // First half are inputs
    auto output_shape = hybrid_net_engine->getTensorShape(hybrid_net_engine->getIOTensorName(output_idx));
    
    // Expected shape: [1, num_keypoints, 3] for 3D coordinates
    // and [1, num_keypoints] for confidences
    int batch_size = output_shape.d[0];
    int num_keypoints = output_shape.d[1];
    
    // Parse 3D coordinates (first output)
    float* coords_data = static_cast<float*>(hybrid_net_host_ptrs[0]);
    
    // Parse confidences (second output, if available)
    float* conf_data = nullptr;
    if (hybrid_net_host_ptrs.size() > 1) {
        conf_data = static_cast<float*>(hybrid_net_host_ptrs[1]);
    }
    
    for (int kp = 0; kp < num_keypoints; kp++) {
        // Extract 3D coordinates
        float x = coords_data[kp * 3 + 0];
        float y = coords_data[kp * 3 + 1];
        float z = coords_data[kp * 3 + 2];
        
        cv::Point3f keypoint_3d(x, y, z);
        
        // Extract confidence
        float confidence = 1.0f; // Default confidence
        if (conf_data) {
            confidence = conf_data[kp];
        }
        
        // Apply confidence threshold
        if (confidence > 0.5f) {
            pose_3d_results.push_back(keypoint_3d);
            pose_3d_confidences.push_back(confidence);
        } else {
            pose_3d_results.push_back(cv::Point3f(0, 0, 0)); // Invalid detection
            pose_3d_confidences.push_back(0.0f);
        }
    }
}

void JarvisPoseDetector::triangulate_3d_center() {
    // Placeholder implementation for 3D center triangulation
    std::cout << "3D center triangulation - placeholder implementation" << std::endl;
}

void JarvisPoseDetector::crop_keypoint_regions() {
    // Placeholder implementation for cropping keypoint regions
    std::cout << "Keypoint region cropping - placeholder implementation" << std::endl;
}

void JarvisPoseDetector::project_3d_to_2d(const cv::Point3f &point_3d, int camera_id, cv::Point2f &point_2d) {
    // Placeholder implementation for 3D to 2D projection
    std::cout << "3D to 2D projection - placeholder implementation" << std::endl;
}

// Getter methods
void JarvisPoseDetector::get_center_results(std::vector<cv::Point2f> &centers, std::vector<float> &confidences) {
    centers = center_results;
    confidences = center_confidences;
}

void JarvisPoseDetector::get_keypoint_results(std::vector<cv::Point2f> &keypoints, std::vector<float> &confidences) {
    keypoints = keypoint_results;
    confidences = keypoint_confidences;
}

void JarvisPoseDetector::get_3d_center_result(cv::Point3f &center_3d, float &confidence) {
    center_3d = center_3d_result;
    confidence = center_3d_confidence;
}

void JarvisPoseDetector::get_3d_pose_results(std::vector<cv::Point3f> &keypoints_3d, std::vector<float> &confidences) {
    keypoints_3d = pose_3d_results;
    confidences = pose_3d_confidences;
}
