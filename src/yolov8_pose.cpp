#include "yolov8_pose.h"

YOLOv8_pose::YOLOv8_pose(const std::string& engine_file_path)
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

YOLOv8_pose::~YOLOv8_pose()
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

void YOLOv8_pose::make_pipe(bool warmup)
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

void YOLOv8_pose::preprocess_gpu(unsigned char* d_rgb)
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


void YOLOv8_pose::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
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
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image)
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

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}
 
void YOLOv8_pose::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_pose::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto kps_ptr    = row_ptr + 5;

        float score = *scores_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;
            std::vector<float> kps;
            for (int k = 0; k < 4; k++) {
                float kps_x = (*(kps_ptr + 2 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 2 * k + 1) - dh) * ratio;
                // float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x       = clamp(kps_x, 0.f, width);
                kps_y       = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                // kps.push_back(kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(0);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        obj.kps   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8_pose::copy_keypoints_gpu(float* d_points, const std::vector<Object>& objs)
{
    const int num_point = 4;
    float points[8];
    // TODO: draw both the bbox and the keypoints 
    for (auto& obj : objs) {
        auto& kps = obj.kps;
        for (int k = 0; k < num_point ; k++) {
            points[k*2] = kps[k * 2];
            points[k*2+1] = kps[k * 2 +1];
        }
    }
    CHECK(cudaMemcpy(d_points, points, sizeof(float) * 8, cudaMemcpyHostToDevice));
}

void YOLOv8_pose::draw_objects(const cv::Mat&                                image,
                               cv::Mat&                                      res,
                               const std::vector<Object>&                    objs,
                               const std::vector<std::vector<unsigned int>>& SKELETON,
                               const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                               const std::vector<std::vector<unsigned int>>& LIMB_COLORS)
{
    res                 = image.clone();
    const int num_point = 4;
    for (auto& obj : objs) {
        cv::rectangle(res, obj.rect, {0, 0, 255}, 2);

        char text[256];
        sprintf(text, "rat %.1f%%", obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

        auto& kps = obj.kps;

        for (int k = 0; k < num_point ; k++) {
                int   kps_x = std::round(kps[k * 2]);
                int   kps_y = std::round(kps[k * 2 + 1]);
                // float kps_s = kps[k * 3 + 2];
                // if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                // }
        }

        for (int k = 0; k < num_point - 1; k++) {
            if (k < num_point) {
                int   kps_x = std::round(kps[k * 2]);
                int   kps_y = std::round(kps[k * 2 + 1]);
                // float kps_s = kps[k * 3 + 2];
                // if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                // }
            }
            auto& ske    = SKELETON[k];
            int   pos1_x = std::round(kps[(ske[0] - 1) * 2]);
            int   pos1_y = std::round(kps[(ske[0] - 1) * 2 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 2]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 2 + 1]);

            // float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            // float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            // if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            // }
        }
    }
}

