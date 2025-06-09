#include "yolov8_det.h"

YOLOv8::YOLOv8(const std::string& engine_file_path, int width, int height)
{
    img_width = width;
    img_height = height;

    // Initialize stream to nullptr, it will be created in make_pipe if needed
    this->stream = nullptr; 
    // Initialize other member pointers to nullptr
    this->d_temp = nullptr;
    this->d_boarder = nullptr;
    this->d_float = nullptr;
    this->d_planar = nullptr;


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
    // cudaStreamCreate(&this->stream); // Moved to make_pipe for conditional creation
    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        const char* name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(this->engine->getIOTensorName(i));
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        nvinfer1::TensorIOMode ioMode = this->engine->getTensorIOMode(name);
        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setInputShape(name, dims);
        }
        else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            dims         = this->context->getTensorShape(name);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

    auto&    in_binding = this->input_bindings[0];

    inp_h_int = in_binding.dims.d[2];
    inp_w_int = in_binding.dims.d[3];

    const float inp_h  = (float)inp_h_int;
    const float inp_w  = (float)inp_w_int;
    float       img_width_float  = img_width;
    float       img_height_float = img_height;

    float r    = std::min(inp_h / img_height_float, inp_w / img_width_float);
    padw = std::round(img_width_float * r);
    padh = std::round(img_height_float * r);
}

YOLOv8::~YOLOv8()
{
    std::cout << "YOLOv8 destructor called." << std::endl;
    if (this->stream) {
        cudaError_t stream_sync_err = cudaStreamSynchronize(this->stream);
        if (stream_sync_err != cudaSuccess) {
             std::cerr << "YOLOv8 destructor: cudaStreamSynchronize failed. Error: " << cudaGetErrorString(stream_sync_err) << std::endl;
        }
    }

    // 1. Free binding buffers
    for (auto& ptr : this->device_ptrs) {
        if (ptr) {
            cudaError_t free_err = cudaFree(ptr);
            if (free_err != cudaSuccess) {
                std::cerr << "YOLOv8 destructor: cudaFree for device_ptr (binding) failed. Error: " << cudaGetErrorString(free_err) << std::endl;
            }
        }
    }
    this->device_ptrs.clear();

    for (auto& ptr : this->host_ptrs) {
        if (ptr) {
            cudaError_t free_host_err = cudaFreeHost(ptr);
            if (free_host_err != cudaSuccess) {
                std::cerr << "YOLOv8 destructor: cudaFreeHost for host_ptr (binding) failed. Error: " << cudaGetErrorString(free_host_err) << std::endl;
            }
        }
    }
    this->host_ptrs.clear();

    // 2. Free other manually allocated GPU buffers that are class members
    if (this->d_planar) { cudaFree(this->d_planar); this->d_planar = nullptr; }
    if (this->d_float) { cudaFree(this->d_float); this->d_float = nullptr; }
    if (this->d_boarder) { cudaFree(this->d_boarder); this->d_boarder = nullptr; }
    if (this->d_temp) { cudaFree(this->d_temp); this->d_temp = nullptr; }

    // 3. Destroy the CUDA stream
    if (this->stream) {
        cudaError_t stream_destroy_err = cudaStreamDestroy(this->stream);
        if (stream_destroy_err != cudaSuccess) {
             std::cerr << "YOLOv8 destructor: cudaStreamDestroy failed. Error: " << cudaGetErrorString(stream_destroy_err) << std::endl;
        }
        this->stream = nullptr;
    }

    // 4. Destroy TensorRT objects (RAII through delete)
    if (this->context) {
        delete this->context;
        this->context = nullptr;
    }
    if (this->engine) {
        delete this->engine;
        this->engine = nullptr;
    }
    if (this->runtime) {
        delete this->runtime;
        this->runtime = nullptr;
    }
    std::cout << "YOLOv8 destructor finished." << std::endl;
}

void YOLOv8::make_pipe(bool warmup)
{
    // Create stream if it hasn't been created already
    if (this->stream == nullptr) {
        CHECK(cudaStreamCreate(&this->stream));
    }

    // Use class members for these allocations
    CHECK(cudaMalloc((void **)&this->d_temp, padw * padh * 3 * sizeof(unsigned char))); // added sizeof
    CHECK(cudaMalloc((void **)&this->d_boarder, inp_w_int * inp_h_int * 3 * sizeof(unsigned char))); // used inp_h_int, added sizeof
    CHECK(cudaMalloc((void **)&this->d_float, sizeof(float) * inp_w_int * inp_h_int * 3)); // used inp_h_int
    CHECK(cudaMalloc((void **)&this->d_planar, sizeof(float) * inp_w_int * inp_h_int * 3)); // used inp_h_int

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
        this->device_ptrs.push_back(d_ptr); // This correctly adds output device pointers after input ones
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (size_t j = 0; j < this->input_bindings.size(); ++j) { // Iterate through actual input bindings
                size_t size  = this->input_bindings[j].size * this->input_bindings[j].dsize;
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                // Use the correct device_ptrs index for inputs
                CHECK(cudaMemcpyAsync(this->device_ptrs[j], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::preprocess_gpu(unsigned char *d_rgb_from_camera) // Renamed for clarity
{
    const float inp_h  = (float)inp_h_int;
    const float inp_w  = (float)inp_w_int;
    float       current_img_width  = img_width;  // Use member img_width
    float       current_img_height = img_height; // Use member img_height

    float r    = std::min(inp_h / current_img_height, inp_w / current_img_width);
    int   current_padw = std::round(current_img_width * r);
    int   current_padh = std::round(current_img_height * r);


    NppiSize src_img_size;
    src_img_size.width = img_width;   // Use member img_width
    src_img_size.height = img_height; // Use member img_height
    NppiRect src_roi;
    src_roi.x = 0;
    src_roi.y = 0;
    src_roi.width = img_width;    // Use member img_width
    src_roi.height = img_height;  // Use member img_height

    NppiSize resized_output_size;
    resized_output_size.width = current_padw;
    resized_output_size.height = current_padh;
    // ROI for resize output is the full resized image
    NppiRect resized_output_roi = {0, 0, current_padw, current_padh};


    // nppiResize_8u_C3R expects pitch in bytes for source and destination
    const NppStatus npp_result_resize = nppiResize_8u_C3R(d_rgb_from_camera,
                                            img_width * 3 * sizeof(unsigned char), // Source pitch
                                            src_img_size,
                                            src_roi,
                                            this->d_temp, // Using class member
                                            current_padw * 3 * sizeof(unsigned char), // Destination pitch
                                            resized_output_size,
                                            resized_output_roi,
                                            NPPI_INTER_SUPER);
    if (npp_result_resize != NPP_SUCCESS) {
        std::cerr << "Error executing nppiResize_8u_C3R -- code: " << npp_result_resize << std::endl;
    }

    NppiSize final_input_tensor_size; // This is inp_w_int x inp_h_int
    final_input_tensor_size.width = inp_w_int;
    final_input_tensor_size.height = inp_h_int;

    float dw_border = (inp_w - current_padw) / 2.0f;
    float dh_border = (inp_h - current_padh) / 2.0f;
    int top_border    = static_cast<int>(std::round(dh_border - 0.1f));
    int left_border   = static_cast<int>(std::round(dw_border - 0.1f));

    Npp8u border_val[3] = {114, 114, 114};
    // nppiCopyConstBorder_8u_C3R expects pitch in bytes
    const NppStatus npp_result_border = nppiCopyConstBorder_8u_C3R(this->d_temp, // Using class member
                                            current_padw * 3 * sizeof(unsigned char), // Source pitch
                                            resized_output_size,      // Source size (image data in d_temp is this size)
                                            this->d_boarder,          // Using class member
                                            inp_w_int * 3 * sizeof(unsigned char),  // Destination pitch
                                            final_input_tensor_size,  // Destination size (full model input size)
                                            top_border,
                                            left_border,
                                            border_val);

    if (npp_result_border != NPP_SUCCESS) {
        std::cerr << "Error executing nppiCopyConstBorder_8u_C3R -- code: " << npp_result_border << std::endl;
    }

    // Convert to float, normalize, and transpose
    // nppiConvert_8u32f_C3R expects pitch in bytes
    const NppStatus npp_result_to_float = nppiConvert_8u32f_C3R(this->d_boarder, // Using class member
                                            inp_w_int * 3 * sizeof(unsigned char),
                                            this->d_float, // Using class member
                                            inp_w_int * 3 * sizeof(float),
                                            final_input_tensor_size);
    if (npp_result_to_float != NPP_SUCCESS) {
        std::cerr << "Error executing nppiConvert_8u32f_C3R -- code: " << npp_result_to_float << std::endl;
    }
    
    Npp32f norm_scale_factor[3] = {255.0f, 255.0f, 255.0f};
    // nppiDivC_32f_C3IR expects pitch in bytes
    const NppStatus npp_result_normalize = nppiDivC_32f_C3IR(norm_scale_factor,
                                             this->d_float, // Using class member
                                             inp_w_int * 3 * sizeof(float),
                                             final_input_tensor_size);
    if (npp_result_normalize != NPP_SUCCESS) {
        std::cerr << "Error executing nppiDivC_32f_C3IR -- code: " << npp_result_normalize << std::endl;
    }

    // d_planar is already a class member, so this is correct
    float * const planar_output_channels[3] {this->d_planar,
                                          this->d_planar + inp_w_int * inp_h_int,
                                          this->d_planar + (inp_w_int * inp_h_int * 2)};
    // nppiCopy_32f_C3P3R expects pitch in bytes
    const NppStatus npp_result_transpose = nppiCopy_32f_C3P3R(this->d_float, // Using class member
                                              inp_w_int * 3 * sizeof(float),
                                              planar_output_channels,
                                              inp_w_int * sizeof(float), // Pitch for each plane
                                              final_input_tensor_size);
    if (npp_result_transpose != NPP_SUCCESS) {
        std::cerr << "Error executing nppiCopy_32f_C3P3R -- code: " << npp_result_transpose << std::endl;
    }

    this->pparam.ratio  = 1.0f / r; // Corrected ratio calculation
    this->pparam.dw     = dw_border;
    this->pparam.dh     = dh_border;
    this->pparam.height = current_img_height; // Original image height before letterboxing
    this->pparam.width  = current_img_width;  // Original image width before letterboxing


    const char* binding_name  = this->engine->getIOTensorName(0); // Assuming first binding is input
    // setInputShape might not be needed if profile shapes are fixed and kMAX is always used
    // However, if dynamic shapes are involved, it might be necessary.
    // For now, assuming it's okay to set it, or that it doesn't harm if already set.
    this->context->setInputShape(binding_name, nvinfer1::Dims{4, {1, 3, inp_h_int, inp_w_int}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], this->d_planar, inp_h_int * inp_w_int * 3 * sizeof(float), cudaMemcpyDeviceToDevice, this->stream));
}


void YOLOv8::infer()
{
    for (int32_t i = 0; i < this->num_bindings; ++i) // Use num_bindings
    {
        auto const name = this->engine->getIOTensorName(i);
        this->context->setTensorAddress(name, this->device_ptrs[i]);
    }

    this->context->enqueueV3(this->stream);

    // Correctly iterate only through output bindings for copying back to host
    for (int i = 0; i < this->num_outputs; ++i) {
        // Output bindings are stored in device_ptrs after input bindings
        size_t output_binding_index_in_device_ptrs = this->num_inputs + i;
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], // host_ptrs stores output host buffers
            this->device_ptrs[output_binding_index_in_device_ptrs], // Corresponding device buffer for this output
            osize,
            cudaMemcpyDeviceToHost,
            this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    // Ensure host_ptrs indices are correct based on the order they were added
    // Assuming order: num_dets, boxes, scores, labels
    int* num_dets = static_cast<int*>(this->host_ptrs[0]);
    float* boxes    = static_cast<float*>(this->host_ptrs[1]);
    float* scores   = static_cast<float*>(this->host_ptrs[2]);
    int* labels   = static_cast<int*>(this->host_ptrs[3]);

    // Use pparam members correctly as set in preprocess_gpu
    float& dw_letterbox    = this->pparam.dw;
    float& dh_letterbox    = this->pparam.dh;
    float& original_img_w = this->pparam.width;  // Original image width passed to preprocess
    float& original_img_h = this->pparam.height; // Original image height passed to preprocess
    float& inv_ratio       = this->pparam.ratio;  // This should be 1.0f / r as calculated in preprocess

    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4; // Each box has 4 values (x0, y0, x1, y1)

        // Coordinates from the model are relative to the letterboxed input (inp_w x inp_h)
        float x0_letterboxed = *ptr++;
        float y0_letterboxed = *ptr++;
        float x1_letterboxed = *ptr++;
        float y1_letterboxed = *ptr;

        // Remove padding
        float x0_resized = x0_letterboxed - dw_letterbox;
        float y0_resized = y0_letterboxed - dh_letterbox;
        float x1_resized = x1_letterboxed - dw_letterbox;
        float y1_resized = y1_letterboxed - dh_letterbox;

        // Scale back to original image dimensions
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
    // This function seems to prepare points for drawing a bounding box, not keypoints.
    // It takes the first object's rectangle and calculates its 4 corner points.
    float h_points[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Initialize to avoid using uninitialized memory if objs is empty

    if (!objs.empty()) {
        const auto& obj = objs[0]; // Consider only the first detected object
        h_points[0] = obj.rect.x;
        h_points[1] = obj.rect.y;
        h_points[2] = obj.rect.x + obj.rect.width; // Top-right x
        h_points[3] = obj.rect.y;                  // Top-right y
        h_points[4] = obj.rect.x + obj.rect.width; // Bottom-right x
        h_points[5] = obj.rect.y + obj.rect.height;// Bottom-right y
        h_points[6] = obj.rect.x;                  // Bottom-left x
        h_points[7] = obj.rect.y + obj.rect.height;// Bottom-left y
    }
    // Using default stream (0) for this synchronous copy. Ensure this is intended.
    // If d_points is used in operations on this->stream, consider using cudaMemcpyAsync with this->stream.
    CHECK(cudaMemcpy(d_points, h_points, sizeof(float) * 8, cudaMemcpyHostToDevice));
}

// draw_objects remains the same as it's a static utility function using OpenCV
void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

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