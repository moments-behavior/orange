#include "kernel.cuh"
#include "opencv2/opencv.hpp"
#include "yolov8_det.h"
#include <string> // for std::stoi

const std::vector<std::string> CLASS_NAMES = {"rat"};
const std::vector<std::vector<unsigned int>> COLORS = {{255, 0, 255}};

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s [engine_path] [video_path] [gpu_id]\n",
                argv[0]);
        return -1;
    }

    int device_id = std::stoi(argv[3]);
    // cuda:0
    cudaSetDevice(device_id);

    const std::string engine_file_path{argv[1]};
    const std::string input_video{argv[2]};

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

    printf("YOLO initialization...\n");

    YOLOv8 *yolov8 =
        new YOLOv8(engine_file_path, camera_width, camera_height, true, 0);
    yolov8->make_pipe(true);

    cudaMalloc((void **)&d_points, sizeof(float) * 8);
    cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
    CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8,
                     cudaMemcpyHostToDevice));

    int frame_size = camera_width * camera_height * 3;
    CHECK(cudaMalloc((void **)&d_frame, frame_size));
    std::vector<Bbox> objs;
    unsigned char *frame_draw =
        (unsigned char *)malloc(frame_size * sizeof(unsigned char));

    cv::Mat view;
    cv::Mat final_view;
    while (cap.read(image)) {

        auto start = std::chrono::high_resolution_clock::now();
        CHECK(cudaMemcpy(d_frame, (uint8_t *)image.data, frame_size,
                         cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        std::cout << "copy frame from cpu to gpu:  " << elapsed.count() << " ms"
                  << std::endl;

        start = std::chrono::high_resolution_clock::now();
        yolov8->preprocess_gpu(d_frame);
        yolov8->infer();
        yolov8->postprocess(objs);
        yolov8->copy_keypoints_gpu(d_points, objs);
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        elapsed = stop - start;
        std::cout << "yolo pre/infer/post time:  " << elapsed.count() << " ms"
                  << std::endl;

        gpu_draw_rat_pose(d_frame, camera_width, camera_height, d_points,
                          d_skeleton, yolov8->stream, 3);
        // copy frame back for opencv visualization
        cudaMemcpy2D(frame_draw, camera_width * 3, d_frame, camera_width * 3,
                     camera_width * 3, camera_height, cudaMemcpyDeviceToHost);
        view = cv::Mat(camera_width * camera_height * 3, 1, CV_8U, frame_draw)
                   .reshape(3, camera_height);
        // yolov8->draw_objects(image, view, objs, CLASS_NAMES, COLORS);
        float r = 0.5;
        int output_w = std::round(camera_width * r);
        int output_h = std::round(camera_height * r);
        cv::resize(view, final_view, cv::Size(output_w, output_h));

        cv::imshow(engine_file_path.c_str(), final_view);
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }
}
