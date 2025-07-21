#include "yolov8.h"
#include "utils.h"
#include <algorithm>
#include <math.h>
#include <npp.h>
#include <nvToolsExt.h>
#include <unordered_map>

YOLOv8::YOLOv8(const std::string &engine_file_path, int width, int height,
               cudaStream_t stream, unsigned char *d_input_image,
               const NppStreamContext &npp_ctx)
    : stream(stream), d_input_image(d_input_image), npp_ctx_(npp_ctx) {

    d_temp = d_boarder = nullptr;
    d_float = d_planar = nullptr;
    inference_graph = nullptr;
    inference_graph_exec = nullptr;

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
        this->context = nullptr;
    }

    if (this->engine) {
        delete this->engine; // Use delete to call the destructor for `engine`.
        this->engine = nullptr;
    }

    if (this->runtime) {
        delete this
            ->runtime; // Use delete to call the destructor for `runtime`.
    }

    if (inference_graph_exec) {
        cudaGraphExecDestroy(inference_graph_exec);
    }
    if (inference_graph) {
        cudaGraphDestroy(inference_graph);
    }

    for (auto &ptr : this->device_ptrs) {
        if (ptr)
            CHECK(cudaFreeAsync(ptr, this->stream));
    }

    for (auto &ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }

    device_ptrs.clear();
    host_ptrs.clear();

    if (d_temp)
        CHECK(cudaFreeAsync(d_temp, stream));
    if (d_boarder)
        CHECK(cudaFreeAsync(d_boarder, stream));
    if (d_float)
        CHECK(cudaFreeAsync(d_float, stream));
    if (d_planar)
        CHECK(cudaFreeAsync(d_planar, stream));
    stream = nullptr;
}

void YOLOv8::make_pipe(bool graph_capture) {
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
        void *d_ptr = nullptr;
        void *h_ptr = nullptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (graph_capture) {
        // Step 1: one warmup pass to trigger JIT and plugin init
        this->preprocess_gpu();
        this->infer_capture_only(); // no sync!

        // Step 2: capture graph
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;

        CHECK(cudaStreamSynchronize(this->stream)); // ensure warmup is done
        CHECK(
            cudaStreamBeginCapture(this->stream, cudaStreamCaptureModeGlobal));
        this->preprocess_gpu();
        this->infer_capture_only(); // no sync inside this!

        CHECK(cudaStreamEndCapture(this->stream, &graph));
        CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        // Save for reuse
        this->inference_graph = graph;
        this->inference_graph_exec = graph_exec;
        this->graph_captured = true;

        CHECK(cudaStreamSynchronize(this->stream));
        printf("CUDA Graph captured with preprocess + inference.\n");
    } else {
        this->preprocess_gpu();
        this->infer();
        this->graph_captured = false;
    }
}

void YOLOv8::preprocess_gpu() {
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
    const NppStatus npp_result = nppiResize_8u_C3R_Ctx(
        d_input_image, img_width * sizeof(uchar3), img_size, roi, d_temp,
        inp_w_int * sizeof(uchar3), output_resize_size, output_roi,
        NPPI_INTER_SUPER, npp_ctx_);
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
    const NppStatus npp_result2 = nppiCopyConstBorder_8u_C3R_Ctx(
        d_temp, inp_w_int * sizeof(uchar3), output_resize_size, d_boarder,
        inp_w_int * sizeof(uchar3), boarder_size, top, left, boarder_color,
        npp_ctx_);

    if (npp_result2 != NPP_SUCCESS) {
        std::cerr << "Error executing CopyConstBoarder -- code: " << npp_result2
                  << std::endl;
    }

    // blobImageNPP: 1. convert to float: nppiConvert_8u32f_C3R; 2. normalize,
    // nppiDivC_32f_C3IR; 3. transpose: nppiCopy_32f_C3P3R
    const NppStatus npp_result3 = nppiConvert_8u32f_C3R_Ctx(
        d_boarder, inp_w_int * sizeof(uchar3), d_float,
        inp_w_int * sizeof(float3), boarder_size, npp_ctx_);
    if (npp_result3 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result3
                  << std::endl;
    }

    Npp32f scale_factor[3] = {255.0f, 255.0f, 255.0f};

    const NppStatus npp_result4 =
        nppiDivC_32f_C3IR_Ctx(scale_factor, d_float, inp_w_int * sizeof(float3),
                              boarder_size, npp_ctx_);
    if (npp_result4 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result4
                  << std::endl;
    }

    float *const inputArr[3]{d_planar, d_planar + inp_w_int * inp_w_int,
                             d_planar + (inp_w_int * inp_w_int * 2)};
    const NppStatus npp_result5 = nppiCopy_32f_C3P3R_Ctx(
        d_float, inp_w_int * sizeof(float3), inputArr,
        inp_w_int * sizeof(float), boarder_size, npp_ctx_);
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

void YOLOv8::infer_capture_only() {

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
}

void YOLOv8::postprocess(std::vector<Object> &objs) {
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
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        objs.push_back(obj);
    }
}

void YOLOv8::postprocess_kp(std::vector<Object> &objs, float score_thres,
                            float iou_thres, int topk) {
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors = this->output_bindings[0].dims.d[2];

    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
                             static_cast<float *>(this->host_ptrs[0]));
    output = output.t();

    // cv::Mat column = output.col(5); // extract column 1 (2nd column)
    // double minVal, maxVal;
    // cv::minMaxLoc(column, &minVal, &maxVal);
    // std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;
    // for (int j = 0; j < output.cols; ++j) {
    //     std::cout << row_ptr[j] << " ";
    // }
    // std::cout << std::endl;

    for (int i = 0; i < num_anchors; i++) {

        auto row_ptr = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;

        float score_class0 = *(row_ptr + 4);
        float score_class1 = *(row_ptr + 5);
        float score;
        int class_id;
        if (score_class1 > score_class0) {
            score = score_class1;
            class_id = 1;
        } else {
            score = score_class0;
            class_id = 0;
        }

        auto kps_ptr = row_ptr + 6;

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
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;
            std::vector<float> kps;
            for (int k = 0; k < 4; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x = clamp(kps_x, 0.f, width);
                kps_y = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(class_id);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

    // auto t_start = std::chrono::high_resolution_clock::now();

    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres,
                             indices);

    // auto t_end = std::chrono::high_resolution_clock::now();
    // double duration_ms =
    //     std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // std::cout << "NMSBoxesBatched took " << duration_ms << " ms" <<
    // std::endl;

    // Step 1: group indices by class label
    std::unordered_map<int, std::vector<int>> class_to_indices;

    for (int idx : indices) {
        int cls = labels[idx];
        class_to_indices[cls].push_back(idx);
    }

    // Step 2: for each class, sort and keep top-k
    for (auto &[cls, idxs] : class_to_indices) {
        // Sort by descending confidence
        std::sort(idxs.begin(), idxs.end(),
                  [&](int a, int b) { return scores[a] > scores[b]; });

        // Step 3: push top-k detections for this class
        int count = std::min(topk, static_cast<int>(idxs.size()));
        for (int j = 0; j < count; ++j) {
            int i = idxs[j];
            Object obj;
            obj.rect = bboxes[i];
            obj.prob = scores[i];
            obj.label = labels[i];
            obj.kps = kpss[i];
            objs.push_back(obj);
        }
    }
}

void YOLOv8::copy_keypoints_gpu(float *d_points,
                                const std::vector<Object> &objs) {
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
    CHECK(cudaMemcpyAsync(d_points, points, sizeof(float) * 8,
                          cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::copy_keypoints_gpu(float *d_points, const Object &obj) {
    float points[8] = {0};

    points[0] = obj.rect.x;
    points[1] = obj.rect.y;

    points[2] = obj.rect.x;
    points[3] = obj.rect.y + obj.rect.height;

    points[4] = obj.rect.x + obj.rect.width;
    points[5] = obj.rect.y + obj.rect.height;

    points[6] = obj.rect.x + obj.rect.width;
    points[7] = obj.rect.y;

    CHECK(cudaMemcpyAsync(d_points, points, sizeof(float) * 8,
                          cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::draw_objects(
    const cv::Mat &image, cv::Mat &res, const std::vector<Object> &objs,
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

void YOLOv8::draw_objects_kp(
    const cv::Mat &image, cv::Mat &res, const std::vector<Object> &objs,
    const std::vector<std::vector<unsigned int>> &SKELETON,
    const std::vector<std::vector<unsigned int>> &KPS_COLORS,
    const std::vector<std::vector<unsigned int>> &LIMB_COLORS) {
    res = image.clone();
    const int num_point = 4;
    for (auto &obj : objs) {
        cv::Scalar obj_color; // Use this for drawing
        char text[256];
        if (obj.label == 0) {
            obj_color = cv::Scalar(0, 0, 255);
            sprintf(text, "rat %.1f%%", obj.prob * 100);
        } else {
            obj_color = cv::Scalar(255, 0, 0);
            sprintf(text, "ball %.1f%%", obj.prob * 100);
        }

        cv::rectangle(res, obj.rect, obj_color, 2);

        int baseLine = 0;
        cv::Size label_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
            res, cv::Rect(x, y, label_size.width, label_size.height + baseLine),
            obj_color, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

        auto &kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++) {
            if (k < num_point) {
                int kps_x = std::round(kps[k * 3]);
                int kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(
                        KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto &ske = SKELETON[k];
            int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(
                    LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color,
                         2);
            }
        }
    }
}
