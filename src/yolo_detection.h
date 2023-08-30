#ifndef ORANGE_YOLO
#define ORANGE_YOLO
#include <opencv2/dnn.hpp>
#include <iostream>
#include <condition_variable>
#include <iomanip>
#include <thread> 
#include <string>
#include <fstream>


struct yolo_sync {
    bool new_frame;
    bool detect_ready; 
};

struct yolo_param {
    float conf_threshold;
    float nma_threshold;
    int size_class_list;
    std::vector<std::string> class_names;
    yolo_param(): conf_threshold(0.50), nma_threshold(0.45) {}
};

struct center_triangulate {
    cv::Point2f point_2d;            // center location 
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    cv::Point3f point_3d;        // center of object (in Meters) if ZED 3D Camera is used
};

void read_yolo_labels(std::string label_names_file, yolo_param* post_setting)
{
    std::ifstream ifs(label_names_file);
    std::vector<std::string> class_list;
    std::string line;    

    while (std::getline(ifs, line))
    {
        class_list.push_back(line);
    }
    post_setting->size_class_list = class_list.size();
    post_setting->class_names = class_list;
}


void yolo_detection(cv::dnn::Net yolo_net, yolo_param* post_setting, PictureBuffer* display_buffer, int camera_id, std::vector<std::vector<cv::Rect>>& yolo_boxes, std::vector<std::vector<std::string>>& yolo_labels, std::vector<std::vector<int>>& yolo_classid)
{

    cv::Mat image = cv::Mat(3208 * 2200 * 3, 1, CV_8U, display_buffer->frame).reshape(3, 2200);
    double x_factor = image.cols / 640.0;
    double y_factor = image.rows / 640.0;
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1./255.,  cv::Size(640, 640),  cv::Scalar(), true, false);
    yolo_net.setInput(blob);
    std::vector<cv::Mat> outs;
    yolo_net.forward(outs, yolo_net.getUnconnectedOutLayersNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    const int rows = 25200;
    float *data = (float *)outs[0].data;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence > post_setting->conf_threshold)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, post_setting->size_class_list, CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > post_setting->conf_threshold)
            {
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                confidences.push_back((float)confidence);
                classIds.push_back(class_id.x);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 7;
    }

    std::vector<int> indices;
    std::vector<cv::Rect> final_boxes;
    std::vector<std::string> final_labels;
    std::vector<int> final_class_ids;
    cv::dnn::NMSBoxes(boxes, confidences, post_setting->conf_threshold, post_setting->nma_threshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        final_boxes.push_back(box);
        std::stringstream stream;
        stream << " " << std::fixed << std::setprecision(2) << confidences[idx];
        std::string s = post_setting->class_names[classIds[idx]] + stream.str();
        final_labels.push_back(s);
        final_class_ids.push_back((int)classIds[idx]);
    }

    yolo_boxes.at(camera_id) = final_boxes;
    yolo_labels.at(camera_id) = final_labels;
    yolo_classid.at(camera_id) = final_class_ids;
}

#endif
