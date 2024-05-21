#include "yolov8_det.h"

YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLOv8::make_pipe(bool warmup)
{
    // allocate device resources for initialization
    cudaMalloc((void **)&d_temp, 640*439*3);
    cudaMalloc((void **)&d_boarder, 640*640*3);
    cudaMalloc((void **)&d_float, sizeof(float) * 640 * 640 *3);
    cudaMalloc((void **)&d_planar, sizeof(float) * 640 * 640 *3);

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::preprocess_gpu(unsigned char* d_rgb)
{
    const float inp_h  = 640;
    const float inp_w  = 640;
    float       width  = 3208;
    float       height = 2200;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);


    // npp resize, todo: check if resize needed
    NppiSize img_size;
    img_size.width = 3208;
    img_size.height = 2200;
    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = 3208;
    roi.height = 2200;

    NppiSize output_resize_size;
    output_resize_size.width = 640;
    output_resize_size.height = 439;
    NppiRect output_roi;
    output_roi.x = 0;
    output_roi.y = 0;
    output_roi.width = 640;
    output_roi.height = 439;


    const NppStatus npp_result = nppiResize_8u_C3R(d_rgb, 3208 * sizeof(uchar3), img_size, roi, d_temp, 640 * sizeof(uchar3), output_resize_size, output_roi, NPPI_INTER_SUPER);
    if (npp_result != NPP_SUCCESS) {
        std::cerr << "Error executing Resize -- code: " << npp_result << std::endl;
    }

    // make boarder
    NppiSize boarder_size;
    boarder_size.width = 640;
    boarder_size.height = 640;

    Npp8u boarder_color[3] = {114, 114, 114};
    const NppStatus npp_result2 = nppiCopyConstBorder_8u_C3R(d_temp, 640 * sizeof(uchar3), output_resize_size, d_boarder, 640 * sizeof(uchar3), boarder_size, 100, 0, boarder_color);
    if (npp_result2 != NPP_SUCCESS) {
        std::cerr << "Error executing CopyConstBoarder -- code: " << npp_result2 << std::endl;
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;


    // blobImageNPP: 1. convert to float: nppiConvert_8u32f_C3R; 2. normalize, nppiDivC_32f_C3IR; 3. transpose: nppiCopy_32f_C3P3R
    const NppStatus npp_result3 = nppiConvert_8u32f_C3R(d_boarder, 640 * sizeof(uchar3), d_float, 640 * sizeof(float3), boarder_size);
    if (npp_result3 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result3 << std::endl;
    }
    
    Npp32f scale_factor[3] = {255.0f, 255.0f, 255.0f};

    const NppStatus npp_result4 = nppiDivC_32f_C3IR(scale_factor, d_float, 640 * sizeof(float3), boarder_size);
    if (npp_result4 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result4 << std::endl;
    }

     float * const inputArr[3] {d_planar, d_planar + 640 * 640, d_planar + (640 * 640 * 2)};
     const NppStatus npp_result5 = nppiCopy_32f_C3P3R(d_float, 640 * sizeof(float3), inputArr, 640 * sizeof(float), boarder_size);
     if (npp_result5 != NPP_SUCCESS) {
         std::cerr << "Error executing convert to plannar -- code: " << npp_result5 << std::endl;
     }

    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, 640, 640}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], d_planar, 640*640*sizeof(float3), cudaMemcpyDeviceToDevice, this->stream));
}


void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    int*  num_dets = static_cast<int*>(this->host_ptrs[0]);
    auto* boxes    = static_cast<float*>(this->host_ptrs[1]);
    auto* scores   = static_cast<float*>(this->host_ptrs[2]);
    int*  labels   = static_cast<int*>(this->host_ptrs[3]);
    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;
    //std::cout << "number of detections = " << num_dets[0] << std::endl;
    Object obj;
        
    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;
        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;
        
        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        obj.rect.x      = x0;
        obj.rect.y      = y0;
        obj.rect.width  = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob        = *(scores + i);
        obj.label       = *(labels + i);
        objs.push_back(obj);
    }
}


void YOLOv8::copy_keypoints_gpu(float* d_points, const std::vector<Object>& objs)
{
    const int num_point = 4;
    float points[10] = {0};
    // TODO: draw both the bbox and the keypoints 
    //std::cout << "Number of elements in objs: " << objs.size() << std::endl;

    for (auto& obj : objs) {
        points[0] = obj.rect.x;
        points[1] = obj.rect.y;

        points[2] = obj.rect.x;
        points[3] = obj.rect.y + obj.rect.height;
        
        points[4] = obj.rect.x + obj.rect.width;
        points[5] = obj.rect.y + obj.rect.height;

        points[6] = obj.rect.x + obj.rect.width;
        points[7] = obj.rect.y;

        points[8] = obj.rect.x;    
        points[9] = obj.rect.y;    
        CHECK(cudaMemcpy(d_points, points, sizeof(float) * 10, cudaMemcpyHostToDevice));
    }
}

void YOLOv8::copy_centroid_gpu(float* d_centroid, const std::vector<Object>& objs)
{
    const int num_point = 4;
    float points[2] = {0};
    
    for (auto& obj : objs) {
        points[0] = obj.rect.x + (obj.rect.width / 2);
        points[1] = obj.rect.y + (obj.rect.height / 2);
        CHECK(cudaMemcpy(d_centroid, points, sizeof(float) * 2, cudaMemcpyHostToDevice));
    }
    
}