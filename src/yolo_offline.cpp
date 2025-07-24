#include "kernel.cuh"
#include "opencv2/opencv.hpp"
#include "utils.h"
#include "yolov8.h"
#include <nvToolsExt.h>
#include <string> // for std::stoi

const std::vector<std::string> CLASS_NAMES = {"rat"};
const std::vector<std::vector<unsigned int>> COLORS = {{255, 0, 255}};

const std::vector<std::vector<unsigned int>> KPS_COLORS = {
    {0, 255, 0}, {0, 255, 0}, {255, 128, 0}, {51, 153, 255},
    {0, 255, 0}, {0, 255, 0}, {255, 128, 0}, {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {
    {1, 3}, {2, 3}, {3, 4}, {1, 3}, {2, 3}, {3, 4}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
    {51, 153, 255}, {255, 51, 255}, {0, 255, 0},
    {51, 153, 255}, {255, 51, 255}, {0, 255, 0}};

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,
                "Usage: %s [engine_path] [video_path] [gpu_id] [det/pose]\n",
                argv[0]);
        return -1;
    }

    int device_id = std::stoi(argv[3]);
    // cuda:0
    cudaSetDevice(device_id);

    const std::string engine_file_path{argv[1]};
    const std::string input_video{argv[2]};
    std::string mode = argv[4];
    if (mode != "det" && mode != "pose") {
        std::cerr << "Invalid mode. Use 'det' or 'pose'.\n";
        return -1;
    }

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Cannot open %s\n", input_video.c_str());
        return -1;
    }

    // Get video frame width and height
    int camera_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int camera_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Video Width: " << camera_width << std::endl;
    std::cout << "Video Height: " << camera_height << std::endl;

    unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // box

    float *d_points;
    unsigned int *d_skeleton;
    unsigned char *d_frame;
    // get input size from the video
    cv::Mat image;

    // for pose only
    int topk = 1;
    float score_thres = 0.2f;
    float iou_thres = 0.2f;

    printf("YOLO initialization...\n");
    int frame_size = camera_width * camera_height * 3;
    CHECK(cudaMalloc((void **)&d_frame, frame_size));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    NppStreamContext npp_ctx = make_npp_stream_context(device_id, stream);
    YOLOv8 *yolov8 = new YOLOv8(engine_file_path, camera_width, camera_height,
                                stream, d_frame, npp_ctx);
    yolov8->make_pipe(false);

    cudaMalloc((void **)&d_points, sizeof(float) * 8);
    cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
    CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8,
                     cudaMemcpyHostToDevice));

    std::vector<Object> objs;
    unsigned char *frame_draw =
        (unsigned char *)malloc(frame_size * sizeof(unsigned char));

    cv::Mat view;
    cv::Mat final_view;
    bool paused = false;
    while (true) {
        if (!paused) {
            // Read and process next frame
            cap >> image;
            if (image.empty())
                break;

            auto start = std::chrono::high_resolution_clock::now();
            CHECK(cudaMemcpy(d_frame, (uint8_t *)image.data, frame_size,
                             cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = stop - start;
            std::cout << "copy frame from cpu to gpu:  " << elapsed.count()
                      << " ms" << std::endl;

            start = std::chrono::high_resolution_clock::now();

            nvtxRangePush("pre-infer");
            if (yolov8->graph_captured) {
                CHECK(cudaGraphLaunch(yolov8->inference_graph_exec, stream));
                CHECK(cudaStreamSynchronize(stream));
            } else {
                yolov8->preprocess_gpu();
                yolov8->infer(); // it sync gpu with cpu here
            }
            nvtxRangePop();

            nvtxRangePush("post");
            if (mode == "det")
                yolov8->postprocess(objs);
            else {
                yolov8->postprocess_kp(objs, score_thres, iou_thres, topk);
            }
            nvtxRangePop();

            // gpu drawing
            // yolov8->copy_keypoints_gpu(d_points, objs);
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
            elapsed = stop - start;
            std::cout << "yolo pre/infer/post time:  " << elapsed.count()
                      << " ms" << std::endl;

            // gpu_draw_rat_pose(d_frame, camera_width, camera_height, d_points,
            //                   d_skeleton, yolov8->stream, 3);
            // // copy frame back for opencv visualization
            // cudaMemcpy2D(frame_draw, camera_width * 3, d_frame, camera_width
            // * 3,
            //              camera_width * 3, camera_height,
            //              cudaMemcpyDeviceToHost);
            // view = cv::Mat(camera_width * camera_height * 3, 1, CV_8U,
            // frame_draw)
            //            .reshape(3, camera_height);
            if (mode == "det") {
                yolov8->draw_objects(image, view, objs, CLASS_NAMES, COLORS);

            } else {
                yolov8->draw_objects_kp(image, view, objs, SKELETON, KPS_COLORS,
                                        LIMB_COLORS);
            }
            float r = 0.5;
            int output_w = std::round(camera_width * r);
            int output_h = std::round(camera_height * r);
            cv::resize(view, final_view, cv::Size(output_w, output_h));

            cv::imshow(engine_file_path.c_str(), final_view);
        }

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 'p') {
            paused = !paused;
        }
    }
}
