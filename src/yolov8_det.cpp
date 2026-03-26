#include "yolov8_det.h"
#include "common.hpp"
#include "utils.h"
#include <cuda_fp16.h>
#include <npp.h>
#include <nvToolsExt.h>
#include <sstream>

namespace {
const char *dtype_to_cstr(nvinfer1::DataType dt) {
    switch (dt) {
    case nvinfer1::DataType::kFLOAT:
        return "kFLOAT";
    case nvinfer1::DataType::kHALF:
        return "kHALF";
    case nvinfer1::DataType::kINT32:
        return "kINT32";
    case nvinfer1::DataType::kINT8:
        return "kINT8";
    case nvinfer1::DataType::kBOOL:
        return "kBOOL";
    default:
        return "UNKNOWN";
    }
}

float read_as_float(const void *ptr, nvinfer1::DataType dt, int idx) {
    switch (dt) {
    case nvinfer1::DataType::kFLOAT:
        return static_cast<const float *>(ptr)[idx];
    case nvinfer1::DataType::kHALF:
        return __half2float(static_cast<const __half *>(ptr)[idx]);
    case nvinfer1::DataType::kINT32:
        return static_cast<float>(static_cast<const int32_t *>(ptr)[idx]);
    case nvinfer1::DataType::kINT8:
        return static_cast<float>(static_cast<const int8_t *>(ptr)[idx]);
    case nvinfer1::DataType::kBOOL:
        return static_cast<const bool *>(ptr)[idx] ? 1.0f : 0.0f;
    default:
        return 0.0f;
    }
}

int read_as_int(const void *ptr, nvinfer1::DataType dt, int idx) {
    switch (dt) {
    case nvinfer1::DataType::kINT32:
        return static_cast<const int32_t *>(ptr)[idx];
    case nvinfer1::DataType::kINT8:
        return static_cast<int>(static_cast<const int8_t *>(ptr)[idx]);
    case nvinfer1::DataType::kBOOL:
        return static_cast<const bool *>(ptr)[idx] ? 1 : 0;
    case nvinfer1::DataType::kFLOAT:
        return static_cast<int>(std::round(static_cast<const float *>(ptr)[idx]));
    case nvinfer1::DataType::kHALF:
        return static_cast<int>(
            std::round(__half2float(static_cast<const __half *>(ptr)[idx])));
    default:
        return 0;
    }
}
} // namespace

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
        binding.dtype = dtype;

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

    std::cout << "YOLOv8 engine: " << engine_file_path << std::endl;
    std::cout << "YOLOv8 I/O tensors: inputs=" << this->num_inputs
              << " outputs=" << this->num_outputs << std::endl;
    for (int i = 0; i < this->num_bindings; ++i) {
        const char *name = this->engine->getIOTensorName(i);
        const bool is_input =
            this->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name);
        nvinfer1::Dims dims = is_input ? this->engine->getProfileShape(
                                            name, 0, nvinfer1::OptProfileSelector::kMAX)
                                       : this->context->getTensorShape(name);
        std::ostringstream dims_ss;
        dims_ss << "[";
        for (int d = 0; d < dims.nbDims; ++d) {
            if (d) dims_ss << "x";
            dims_ss << dims.d[d];
        }
        dims_ss << "]";
        std::cout << "  " << (is_input ? "IN " : "OUT") << " " << name
                  << " shape=" << dims_ss.str()
                  << " dtype=" << dtype_to_cstr(dtype) << std::endl;
    }
    if (this->num_outputs < 4) {
        std::cerr << "YOLOv8 warning: this pipeline expects >=4 outputs "
                     "(num_dets/boxes/scores/labels). Loaded engine has "
                  << this->num_outputs << " outputs." << std::endl;
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

void YOLOv8::postprocess(std::vector<Bbox> &objs) {
    objs.clear();
    if (this->num_outputs <= 0 || this->host_ptrs.empty()) {
        return;
    }

    // Support single-output engines with shape like [1, N, 6]:
    // [x1, y1, x2, y2, score, class]
    if (this->num_outputs == 1 && this->output_bindings.size() == 1) {
        const void *out_ptr = this->host_ptrs[0];
        const auto out_type = this->output_bindings[0].dtype;
        const auto &dims = this->output_bindings[0].dims;
        const int total_vals = static_cast<int>(this->output_bindings[0].size);
        if (total_vals <= 0) {
            return;
        }

        int n = total_vals / 6;
        if (dims.nbDims >= 2 && dims.d[dims.nbDims - 1] == 6 &&
            dims.d[dims.nbDims - 2] > 0) {
            n = dims.d[dims.nbDims - 2];
        }
        if (n <= 0) {
            return;
        }

        auto &dw = this->pparam.dw;
        auto &dh = this->pparam.dh;
        auto &width = this->pparam.width;
        auto &height = this->pparam.height;
        auto &ratio = this->pparam.ratio;

        constexpr float kConfThreshold = 0.25f;
        static int debug_single_prints = 0;
        for (int i = 0; i < n; i++) {
            const float x0_raw = read_as_float(out_ptr, out_type, i * 6 + 0);
            const float y0_raw = read_as_float(out_ptr, out_type, i * 6 + 1);
            const float x1_raw = read_as_float(out_ptr, out_type, i * 6 + 2);
            const float y1_raw = read_as_float(out_ptr, out_type, i * 6 + 3);
            const float score = read_as_float(out_ptr, out_type, i * 6 + 4);
            const int label = static_cast<int>(
                std::round(read_as_float(out_ptr, out_type, i * 6 + 5)));

            if (debug_single_prints < 12 && i < 3) {
                std::cout << "YOLO raw (1xNx6): i=" << i
                          << " score=" << score << " label=" << label
                          << " box=(" << x0_raw << "," << y0_raw << ","
                          << x1_raw << "," << y1_raw << ")" << std::endl;
                debug_single_prints++;
            }

            if (score < kConfThreshold) {
                continue;
            }

            float x0 = (x0_raw - dw) * ratio;
            float y0 = (y0_raw - dh) * ratio;
            float x1 = (x1_raw - dw) * ratio;
            float y1 = (y1_raw - dh) * ratio;

            x0 = clamp(x0, 0.f, width);
            y0 = clamp(y0, 0.f, height);
            x1 = clamp(x1, 0.f, width);
            y1 = clamp(y1, 0.f, height);
            if (x1 <= x0 || y1 <= y0) {
                continue;
            }

            Bbox obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.prob = score;
            obj.label = label;
            objs.push_back(obj);
        }
        return;
    }

    if (this->num_outputs < 4 || this->host_ptrs.size() < 4) {
        return;
    }

    int idx_num_dets = -1;
    int idx_boxes = -1;
    int idx_scores = -1;
    int idx_labels = -1;
    for (int i = 0; i < static_cast<int>(this->output_bindings.size()); ++i) {
        const auto &name = this->output_bindings[i].name;
        if (name == "num_dets")
            idx_num_dets = i;
        else if (name == "bboxes")
            idx_boxes = i;
        else if (name == "scores")
            idx_scores = i;
        else if (name == "labels")
            idx_labels = i;
    }
    if (idx_num_dets < 0 || idx_boxes < 0 || idx_scores < 0 || idx_labels < 0) {
        return;
    }

    const void *num_dets_ptr = this->host_ptrs[idx_num_dets];
    const void *boxes_ptr = this->host_ptrs[idx_boxes];
    const void *scores_ptr = this->host_ptrs[idx_scores];
    const void *labels_ptr = this->host_ptrs[idx_labels];
    const auto num_dets_type = this->output_bindings[idx_num_dets].dtype;
    const auto boxes_type = this->output_bindings[idx_boxes].dtype;
    const auto scores_type = this->output_bindings[idx_scores].dtype;
    const auto labels_type = this->output_bindings[idx_labels].dtype;

    const int n = read_as_int(num_dets_ptr, num_dets_type, 0);
    static int debug_raw_prints = 0;
    if (debug_raw_prints < 40) {
        std::cout << "YOLO raw: num_dets=" << n
                  << " score0=" << read_as_float(scores_ptr, scores_type, 0)
                  << " label0=" << read_as_int(labels_ptr, labels_type, 0)
                  << " box0=("
                  << read_as_float(boxes_ptr, boxes_type, 0) << ","
                  << read_as_float(boxes_ptr, boxes_type, 1) << ","
                  << read_as_float(boxes_ptr, boxes_type, 2) << ","
                  << read_as_float(boxes_ptr, boxes_type, 3) << ")"
                  << std::endl;
        debug_raw_prints++;
    }
    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;
    if (n < 0 || n > 10000) {
        static bool warned_bad_num_dets = false;
        if (!warned_bad_num_dets) {
            std::cerr << "YOLOv8 warning: unexpected num_dets=" << n
                      << " (engine output layout likely mismatched)." << std::endl;
            warned_bad_num_dets = true;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        float x0 = read_as_float(boxes_ptr, boxes_type, i * 4 + 0) - dw;
        float y0 = read_as_float(boxes_ptr, boxes_type, i * 4 + 1) - dh;
        float x1 = read_as_float(boxes_ptr, boxes_type, i * 4 + 2) - dw;
        float y1 = read_as_float(boxes_ptr, boxes_type, i * 4 + 3) - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Bbox obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = read_as_float(scores_ptr, scores_type, i);
        obj.label = read_as_int(labels_ptr, labels_type, i);
        objs.push_back(obj);
    }
}

void YOLOv8::copy_keypoints_gpu(float *d_points,
                                const std::vector<Bbox> &objs) {
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

void YOLOv8::copy_keypoints_gpu(float *d_points, const Bbox &obj) {
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
