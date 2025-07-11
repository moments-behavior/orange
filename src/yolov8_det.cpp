// src/yolov8_det.cpp
#include "yolov8_det.h"
#include <opencv2/opencv.hpp>

void YOLOv8::initialize_plugins() {
    static bool plugins_initialized = false;
    if (!plugins_initialized) {
        Logger logger; // A temporary logger for this one-time initialization.
        initLibNvInferPlugins(&logger, "");
        plugins_initialized = true;
        std::cout << "[YOLOv8] TensorRT plugins initialized." << std::endl;
    }
}

YOLOv8::YOLOv8(const std::string& engine_file_path, int width, int height)
{
    this->stream = nullptr;
    this->d_planar = nullptr;
    this->engine = nullptr;
    this->runtime = nullptr;
    this->context = nullptr;

    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("YOLOv8: Cannot open engine file: " + engine_file_path);
    }
    
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    if (!this->runtime) {
        delete[] trtModelStream;
        throw std::runtime_error("YOLOv8: Failed to create TensorRT runtime");
    }

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    if (!this->engine) {
        delete[] trtModelStream;
        throw std::runtime_error("YOLOv8: Failed to deserialize CUDA engine");
    }
    delete[] trtModelStream;
    
    this->context = this->engine->createExecutionContext();
    if (!this->context) {
        throw std::runtime_error("YOLOv8: Failed to create execution context");
    }

    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        const char* name = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        nvinfer1::TensorIOMode ioMode = this->engine->getTensorIOMode(name);

        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            
            if (!this->context->setInputShape(name, dims)) {
                throw std::runtime_error("YOLOv8: Failed to set input shape in constructor for " + std::string(name));
            }
        }
        else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            dims = this->context->getTensorShape(name);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

    auto& in_binding = this->input_bindings[0];
    inp_h_int = in_binding.dims.d[2];
    inp_w_int = in_binding.dims.d[3];
}

YOLOv8::~YOLOv8()
{
    if (this->stream) {
        cudaStreamSynchronize(this->stream);
    }
    if (this->context) { delete this->context; }
    if (this->engine) { delete this->engine; }
    if (this->runtime) { delete this->runtime; }
    for (size_t i = 0; i < this->device_ptrs.size(); ++i) { if (this->device_ptrs[i]) cudaFree(this->device_ptrs[i]); }
    for (size_t i = 0; i < this->host_ptrs.size(); ++i) { if (this->host_ptrs[i]) cudaFreeHost(this->host_ptrs[i]); }
    if (this->stream) { cudaStreamDestroy(this->stream); }
}

void YOLOv8::make_pipe(bool warmup, int max_width, int max_height)
{
    if (this->stream == nullptr) {
        CHECK(cudaStreamCreate(&this->stream));
    }

    for (size_t i = 0; i < this->input_bindings.size(); ++i) {
        auto& bindings = this->input_bindings[i];
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (size_t i = 0; i < this->output_bindings.size(); ++i) {
        auto& bindings = this->output_bindings[i];
        void* d_ptr;
        void* h_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, bindings.size * bindings.dsize, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (size_t j = 0; j < this->input_bindings.size(); ++j) {
                size_t size = this->input_bindings[j].size * this->input_bindings[j].dsize;
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[j], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
    }
}

void YOLOv8::preprocess_gpu(unsigned char *d_src, int source_width, int source_height, bool is_color)
{
    const float inp_h  = (float)inp_h_int;
    const float inp_w  = (float)inp_w_int;
    float r = std::min(inp_h / (float)source_height, inp_w / (float)source_width);
    
    launch_optimized_yolo_preprocess(
        d_src,
        (float*)this->device_ptrs[0], //directly write to device pointer
        source_width, source_height,
        inp_w_int, inp_h_int,
        is_color,
        this->stream
    );

    this->pparam.ratio  = 1.0f / r;
    this->pparam.dw     = (inp_w - source_width * r) / 2.0f;
    this->pparam.dh     = (inp_h - source_height * r) / 2.0f;
    this->pparam.width  = (float)source_width;
    this->pparam.height = (float)source_height;
}


void YOLOv8::infer()
{
    if (!this->context) {
        throw std::runtime_error("YOLOv8::infer: Execution context is null");
    }
    
    if (!this->stream) {
        throw std::runtime_error("YOLOv8::infer: CUDA stream is null");
    }
    
    for (int32_t i = 0; i < this->num_bindings; ++i) {
        auto const name = this->engine->getIOTensorName(i);
        if (!this->context->setTensorAddress(name, this->device_ptrs[i])) {
            throw std::runtime_error("YOLOv8::infer: Failed to set tensor address for " + std::string(name));
        }
    }

    if (!this->context->enqueueV3(this->stream)) {
        throw std::runtime_error("YOLOv8::infer: enqueueV3 failed");
    }

    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i],
            this->device_ptrs[this->num_inputs + i],
            osize,
            cudaMemcpyDeviceToHost,
            this->stream));
    }
}


void YOLOv8::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    int* num_dets = static_cast<int*>(this->host_ptrs[0]);
    float* boxes    = static_cast<float*>(this->host_ptrs[1]);
    float* scores   = static_cast<float*>(this->host_ptrs[2]);
    int* labels   = static_cast<int*>(this->host_ptrs[3]);

    float& dw_letterbox    = this->pparam.dw;
    float& dh_letterbox    = this->pparam.dh;
    float& original_img_w = this->pparam.width;
    float& original_img_h = this->pparam.height;
    float& inv_ratio       = this->pparam.ratio;

    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        float x0_letterboxed = *ptr++;
        float y0_letterboxed = *ptr++;
        float x1_letterboxed = *ptr++;
        float y1_letterboxed = *ptr;

        float x0_resized = x0_letterboxed - dw_letterbox;
        float y0_resized = y0_letterboxed - dh_letterbox;
        float x1_resized = x1_letterboxed - dw_letterbox;
        float y1_resized = y1_letterboxed - dh_letterbox;

        float x0_original = clamp(x0_resized * inv_ratio, 0.f, original_img_w);
        float y0_original = clamp(y0_resized * inv_ratio, 0.f, original_img_h);
        float x1_original = clamp(x1_resized * inv_ratio, 0.f, original_img_w);
        float y1_original = clamp(y1_resized * inv_ratio, 0.f, original_img_h);

        Object obj;
        obj.rect.x      = x0_original;
        obj.rect.y      = y0_original;
        obj.rect.width  = x1_original - x0_original;
        obj.rect.height = y1_original - y0_original;
        obj.prob        = *(scores + i);
        obj.label       = *(labels + i);
        objs.push_back(obj);
    }
}

void YOLOv8::copy_keypoints_gpu(float* d_points, const std::vector<Object>& objs)
{
    float h_points[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (!objs.empty()) {
        const auto& obj = objs[0];
        h_points[0] = obj.rect.x;
        h_points[1] = obj.rect.y;
        h_points[2] = obj.rect.x + obj.rect.width;
        h_points[3] = obj.rect.y;
        h_points[4] = obj.rect.x + obj.rect.width;
        h_points[5] = obj.rect.y + obj.rect.height;
        h_points[6] = obj.rect.x;
        h_points[7] = obj.rect.y + obj.rect.height;
    }
    CHECK(cudaMemcpy(d_points, h_points, sizeof(float) * 8, cudaMemcpyHostToDevice));
}

void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        
        // Create the cv::Rect on-the-fly from our custom pose::Rect
        cv::Rect cv_rect(static_cast<int>(obj.rect.x), 
                         static_cast<int>(obj.rect.y), 
                         static_cast<int>(obj.rect.width), 
                         static_cast<int>(obj.rect.height));
        cv::rectangle(res, cv_rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}