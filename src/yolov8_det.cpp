#include "yolov8_det.h"
#include "common.hpp"
#include <npp.h>

YOLOv8::YOLOv8(const std::string &engine_file_path, int width, int height,
               bool use_external_stream, cudaStream_t stream) {
    img_width = width;
    img_height = height;

    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
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

    if (use_external_stream) {
        this->stream = stream;
        nppSetStream(this->stream);
        use_external_stream = true;
    } else {
        CHECK(cudaStreamCreate(&this->stream));
        nppSetStream(this->stream);
        use_external_stream = false;
    }

    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        const char *name = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype =
            this->engine->getTensorDataType(this->engine->getIOTensorName(i));
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        nvinfer1::TensorIOMode ioMode = this->engine->getTensorIOMode(name);
        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims = this->engine->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setInputShape(name, dims);
        } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            dims = this->context->getTensorShape(name);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

    auto &in_binding = this->input_bindings[0];

    inp_h_int = in_binding.dims.d[2];
    inp_w_int = in_binding.dims.d[3];

    const float inp_h = (float)inp_h_int;
    const float inp_w = (float)inp_w_int;
    float img_width_float = img_width;
    float img_height_float = img_height;

    float r = std::min(inp_h / img_height_float, inp_w / img_width_float);
    padw = std::round(img_width_float * r);
    padh = std::round(img_height_float * r);
}

YOLOv8::~YOLOv8() {
    // Assuming context, engine, and runtime are all pointers:
    if (this->context) {
        delete this
            ->context; // Use delete to call the destructor for `context`.
    }

    if (this->engine) {
        delete this->engine; // Use delete to call the destructor for `engine`.
    }

    if (this->runtime) {
        delete this
            ->runtime; // Use delete to call the destructor for `runtime`.
    }

    if (!use_external_stream && stream != nullptr) {
        CHECK(cudaStreamDestroy(this->stream));
    }

    for (auto &ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }

    if (d_temp)
        CHECK(cudaFree(d_temp));
    if (d_boarder)
        CHECK(cudaFree(d_boarder));
    if (d_float)
        CHECK(cudaFree(d_float));
    if (d_planar)
        CHECK(cudaFree(d_planar));
}

void YOLOv8::make_pipe(bool warmup) {
    // allocate device resources for initialization
    CHECK(cudaMallocAsync((void **)&d_temp, padw * padh * 3,
                          this->stream)); // assuming width bigger than height
    CHECK(cudaMallocAsync((void **)&d_boarder, inp_w_int * inp_w_int * 3,
                          this->stream));
    CHECK(cudaMallocAsync((void **)&d_float,
                          sizeof(float) * inp_w_int * inp_w_int * 3,
                          this->stream));
    CHECK(cudaMallocAsync((void **)&d_planar,
                          sizeof(float) * inp_w_int * inp_w_int * 3,
                          this->stream));

    for (auto &bindings : this->input_bindings) {
        void *d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize,
                              this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto &bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        // Warm up preprocessing
        unsigned char *dummy_input;
        size_t dummy_size = img_width * img_height * sizeof(uchar3);
        CHECK(cudaMalloc(&dummy_input, dummy_size));

        for (int i = 0; i < 5; ++i) {
            this->preprocess_gpu(dummy_input);
        }

        // Optional: keep or free dummy_input
        cudaFree(dummy_input);

        for (int j = 0; j < 10; ++j) {
            for (size_t i = 0; i < this->input_bindings.size(); ++i) {
                size_t size = input_bindings[i].size * input_bindings[i].dsize;
                void *h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                cudaMemcpyAsync(this->device_ptrs[i], h_ptr, size,
                                cudaMemcpyHostToDevice, this->stream);
                free(h_ptr);
            }
            this->infer(); // launches inference and output copy
        }
        cudaStreamSynchronize(this->stream); // wait for all 10 inferences
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::preprocess_gpu(unsigned char *d_rgb) {
    const float inp_h = (float)inp_h_int;
    const float inp_w = (float)inp_w_int;
    float width = img_width;
    float height = img_height;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    // npp resize, todo: check if resize needed
    NppiSize img_size;
    img_size.width = img_width;
    img_size.height = img_height;
    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = img_width;
    roi.height = img_height;

    NppiSize output_resize_size;
    output_resize_size.width = padw;
    output_resize_size.height = padh;
    NppiRect output_roi;
    output_roi.x = 0;
    output_roi.y = 0;
    output_roi.width = padw;
    output_roi.height = padh;

    // TODO: is input_w_int here correct
    const NppStatus npp_result =
        nppiResize_8u_C3R(d_rgb, img_width * sizeof(uchar3), img_size, roi,
                          d_temp, inp_w_int * sizeof(uchar3),
                          output_resize_size, output_roi, NPPI_INTER_SUPER);
    if (npp_result != NPP_SUCCESS) {
        std::cerr << "Error executing Resize -- code: " << npp_result
                  << std::endl;
    }

    // make boarder
    NppiSize boarder_size;
    boarder_size.width = inp_w_int;
    boarder_size.height = inp_h_int;

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int left = int(std::round(dw - 0.1f));

    Npp8u boarder_color[3] = {114, 114, 114};
    const NppStatus npp_result2 = nppiCopyConstBorder_8u_C3R(
        d_temp, inp_w_int * sizeof(uchar3), output_resize_size, d_boarder,
        inp_w_int * sizeof(uchar3), boarder_size, top, left, boarder_color);

    if (npp_result2 != NPP_SUCCESS) {
        std::cerr << "Error executing CopyConstBoarder -- code: " << npp_result2
                  << std::endl;
    }

    // blobImageNPP: 1. convert to float: nppiConvert_8u32f_C3R; 2. normalize,
    // nppiDivC_32f_C3IR; 3. transpose: nppiCopy_32f_C3P3R
    const NppStatus npp_result3 =
        nppiConvert_8u32f_C3R(d_boarder, inp_w_int * sizeof(uchar3), d_float,
                              inp_w_int * sizeof(float3), boarder_size);
    if (npp_result3 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result3
                  << std::endl;
    }

    Npp32f scale_factor[3] = {255.0f, 255.0f, 255.0f};

    const NppStatus npp_result4 = nppiDivC_32f_C3IR(
        scale_factor, d_float, inp_w_int * sizeof(float3), boarder_size);
    if (npp_result4 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result4
                  << std::endl;
    }

    float *const inputArr[3]{d_planar, d_planar + inp_w_int * inp_w_int,
                             d_planar + (inp_w_int * inp_w_int * 2)};
    const NppStatus npp_result5 =
        nppiCopy_32f_C3P3R(d_float, inp_w_int * sizeof(float3), inputArr,
                           inp_w_int * sizeof(float), boarder_size);
    if (npp_result5 != NPP_SUCCESS) {
        std::cerr << "Error executing convert to plannar -- code: "
                  << npp_result5 << std::endl;
    }

    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;

    const char *name = this->engine->getIOTensorName(0);
    this->context->setInputShape(
        name, nvinfer1::Dims{4, {1, 3, inp_w_int, inp_w_int}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], d_planar,
                          inp_w_int * inp_w_int * sizeof(float3),
                          cudaMemcpyDeviceToDevice, this->stream));
}

// void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
// {
//     const float inp_h  = size.height;
//     const float inp_w  = size.width;
//     float       height = image.rows;
//     float       width  = image.cols;

//     float r    = std::min(inp_h / height, inp_w / width);
//     int   padw = std::round(width * r);
//     int   padh = std::round(height * r);

//     cv::Mat tmp;
//     if ((int)width != padw || (int)height != padh) {
//         cv::resize(image, tmp, cv::Size(padw, padh));
//     }
//     else {
//         tmp = image.clone();
//     }

//     float dw = inp_w - padw;
//     float dh = inp_h - padh;

//     dw /= 2.0f;
//     dh /= 2.0f;
//     int top    = int(std::round(dh - 0.1f));
//     int bottom = int(std::round(dh + 0.1f));
//     int left   = int(std::round(dw - 0.1f));
//     int right  = int(std::round(dw + 0.1f));

//     cv::copyMakeBorder(tmp, tmp, top, bottom, left, right,
//     cv::BORDER_CONSTANT, {114, 114, 114});

//     cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0,
//     0), true, false, CV_32F); this->pparam.ratio  = 1 / r; this->pparam.dw =
//     dw; this->pparam.dh     = dh; this->pparam.height = height;
//     this->pparam.width  = width;
// }

// void YOLOv8::copy_from_Mat(const cv::Mat& image)
// {
//     cv::Mat  nchw;
//     auto&    in_binding = this->input_bindings[0];
//     auto     width      = in_binding.dims.d[3];
//     auto     height     = in_binding.dims.d[2];
//     cv::Size size{width, height};
//     this->letterbox(image, nchw, size);

//     this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height,
//     width}});

//     CHECK(cudaMemcpyAsync(
//         this->device_ptrs[0], nchw.ptr<float>(), nchw.total() *
//         nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
// }

// void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
// {
//     cv::Mat nchw;
//     this->letterbox(image, nchw, size);
//     this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3,
//     size.height, size.width}}); CHECK(cudaMemcpyAsync(
//         this->device_ptrs[0], nchw.ptr<float>(), nchw.total() *
//         nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
// }

void YOLOv8::infer() {

    for (int32_t i = 0, e = this->engine->getNbIOTensors(); i < e; i++) {
        auto const name = this->engine->getIOTensorName(i);
        this->context->setTensorAddress(name, this->device_ptrs[i]);
    }

    this->context->enqueueV3(this->stream);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize =
            this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(this->host_ptrs[i],
                              this->device_ptrs[i + this->num_inputs], osize,
                              cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8::postprocess(std::vector<Bbox> &objs) {
    objs.clear();
    int *num_dets = static_cast<int *>(this->host_ptrs[0]);
    auto *boxes = static_cast<float *>(this->host_ptrs[1]);
    auto *scores = static_cast<float *>(this->host_ptrs[2]);
    int *labels = static_cast<int *>(this->host_ptrs[3]);
    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;
    for (int i = 0; i < num_dets[0]; i++) {
        float *ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Bbox obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        objs.push_back(obj);
    }
}

void YOLOv8::copy_keypoints_gpu(float *d_points,
                                const std::vector<Bbox> &objs) {
    const int num_point = 4;
    float points[8] = {0};
    // TODO: draw both the bbox and the keypoints
    for (auto &obj : objs) {
        points[0] = obj.rect.x;
        points[1] = obj.rect.y;

        points[2] = obj.rect.x;
        points[3] = obj.rect.y + obj.rect.height;

        points[4] = obj.rect.x + obj.rect.width;
        points[5] = obj.rect.y + obj.rect.height;

        points[6] = obj.rect.x + obj.rect.width;
        points[7] = obj.rect.y;
    }
    CHECK(cudaMemcpy(d_points, points, sizeof(float) * 8,
                     cudaMemcpyHostToDevice));
}

void YOLOv8::copy_keypoints_gpu(float *d_points, const Bbox &obj) {
    const int num_point = 4;
    float points[8] = {0};

    points[0] = obj.rect.x;
    points[1] = obj.rect.y;

    points[2] = obj.rect.x;
    points[3] = obj.rect.y + obj.rect.height;

    points[4] = obj.rect.x + obj.rect.width;
    points[5] = obj.rect.y + obj.rect.height;

    points[6] = obj.rect.x + obj.rect.width;
    points[7] = obj.rect.y;

    CHECK(cudaMemcpy(d_points, points, sizeof(float) * 8,
                     cudaMemcpyHostToDevice));
}

void YOLOv8::draw_objects(
    const cv::Mat &image, cv::Mat &res, const std::vector<Bbox> &objs,
    const std::vector<std::string> &CLASS_NAMES,
    const std::vector<std::vector<unsigned int>> &COLORS) {
    res = image.clone();
    for (auto &obj : objs) {
        cv::Scalar color = cv::Scalar(
            COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(),
                obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
            res, cv::Rect(x, y, label_size.width, label_size.height + baseLine),
            {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
