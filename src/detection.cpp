#include "detection.h"

struct cvCamParams {
    int cam_idx;
    cv::Mat K;
    cv::Mat distCoeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projectionMat;
};

float get_color(int c, int x, int max)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}


struct center_triangulate {
    cv::Point2f point_2d;            // center location 
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    cv::Point3f point_3d;        // center of object (in Meters) if ZED 3D Camera is used
};

cv::Mat triangulate_points(std::vector<cv::Point2f> inPts2d, std::vector<cvCamParams*> cvps)
{
    std::vector<cv::Mat> sfmPoints2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output;
    for (int i=0; i<cvps.size(); i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << inPts2d[i].x, inPts2d[i].y);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, cvps[i]->K, cvps[i]->distCoeffs, cv::noArray(), cvps[i]->K);
        sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(cvps[i]->projectionMat);
        
    }

    cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
    output.convertTo(output, CV_32F);
    return output;
}



std::vector<std::vector<center_triangulate>> get_3d_coordinates(std::vector<std::vector<bbox_t>> bounding_boxes, cvCamParams* CamParam, int num_cameras)
{

    // points 
    std::map<unsigned int, std::vector<cv::Point2f>> mapOfObjects;
    std::map<unsigned int, std::vector<cvCamParams*>> mapOfCameras;
    
    // reformat detection data as dictionary 
    for (int cam_idx = 0; cam_idx < bounding_boxes.size(); cam_idx++){
        for (auto &i : bounding_boxes[cam_idx]) {
            if(mapOfObjects.count(i.obj_id) > 0){
                // calcualte center of mass from bounding box 
                float c_x =  float(i.x) + float(i.w)/2.0;
                float c_y =  float(i.y) + float(i.h)/2.0;
                mapOfObjects[i.obj_id].push_back(cv::Point2f(c_x, c_y));
                mapOfCameras[i.obj_id].push_back(CamParam + cam_idx);
            }
            else{
                float c_x =  float(i.x) + float(i.w)/2.0;
                float c_y =  float(i.y) + float(i.h)/2.0;
                std::vector<cv::Point2f> points_per_obj; 
                std::vector<cvCamParams*> camera_per_obj;
                points_per_obj.push_back(cv::Point2f(c_x, c_y));
                camera_per_obj.push_back(CamParam + cam_idx);
                mapOfObjects.insert({i.obj_id, points_per_obj});
                mapOfCameras.insert({i.obj_id, camera_per_obj});
            }
        }
    }


    
    // triangulation
    std::map<unsigned int, cv::Mat> mapOfPoints3D;
    for ( auto it = mapOfObjects.begin(); it != mapOfObjects.end(); ++it)
    {
        std::vector<center_triangulate> single_object_results;

        if(it->second.size() >= 2){
            // triangulate if there are 2 camera detection
            cv::Mat point3d = triangulate_points(it->second, mapOfCameras[it->first]);  
            cv::Point3f triangulate_point = cv::Point3f(point3d.at<float>(0), point3d.at<float>(1), point3d.at<float>(2));
            mapOfPoints3D.insert({it->first, point3d});
        }
    }


    std::vector<std::vector<center_triangulate>> triangulate_res;     // cam, obj, points
    // reproject 
    for (int i=0; i<num_cameras; i++)
    {
        std::vector <center_triangulate> objects_per_camera;
        for (auto it = mapOfPoints3D.begin(); it != mapOfPoints3D.end(); ++it){
            center_triangulate single_object;
            
            single_object.obj_id = it->first;
            single_object.point_3d = cv::Point3f(it->second.at<float>(0), it->second.at<float>(1), it->second.at<float>(2));


            cv::Mat imagePts;
            cv::projectPoints(it->second, CamParam[i].rvec, CamParam[i].tvec, CamParam[i].K, CamParam[i].distCoeffs, imagePts);
            single_object.point_2d = cv::Point2f(imagePts.at<float>(0, 0), imagePts.at<float>(0, 1)); 
            objects_per_camera.push_back(single_object);
        }
        triangulate_res.push_back(objects_per_camera);
    }  
    return triangulate_res;
}


void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, std::vector<center_triangulate> tri_result,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    for (auto &i : result_vec) {

        // cv::Scalar color = obj_id_to_color(i.obj_id);
        int offset = i.obj_id * 123457 % 2;
        float red = get_color(2, offset, 2);
        float green = get_color(1, offset, 2);
        float blue = get_color(0, offset, 2);
        float rgb[3];
        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        cv::Scalar color;
        color.val[0] = red * 256;
        color.val[1] = green * 256;
        color.val[2] = blue * 256;


        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];


            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
        }
    }


    // if (current_det_fps >= 0 && current_cap_fps >= 0) {
    //     std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
    //     putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    // }



    for (auto &i : tri_result) {
        // cv::Scalar color = obj_id_to_color(i.obj_id);
        int offset = i.obj_id * 123457 % 2;
        float red = get_color(2, offset, 2);
        float green = get_color(1, offset, 2);
        float blue = get_color(0, offset, 2);
        float rgb[3];
        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        cv::Scalar color;
        color.val[0] = red * 256;
        color.val[1] = green * 256;
        color.val[2] = blue * 256;
        cv::circle( mat_img, i.point_2d, 8, color, cv::FILLED, cv::LINE_8);      


        
        std::string coords_3d;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << "x:" << i.point_3d.x << "mm y:" << i.point_3d.y << "mm z:" << i.point_3d.z << "mm ";
        coords_3d = ss.str();
        putText(mat_img, coords_3d, cv::Point2f(10, 50+50*i.obj_id), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2);
    }

}



std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}



std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}


cvCamParams LoadCameraParamsFromCSV(std::string csv_filename, int camNum)
{
    std::cout << csv_filename << std::endl;
    cvCamParams cvp;

    std::ifstream fin;
    fin.open(csv_filename);
    if (fin.fail()) throw csv_filename;  // the exception being checked

    std::string line;
    std::string delimeter = ",";
    size_t pos = 0;
    std::string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    std::vector<float> csvCamValues;

    while(!fin.eof()){
        fin >> line;

        while ((pos = line.find(delimeter)) != std::string::npos)
        {
            token = line.substr(0, pos);
            if (lineNum == camNum)
            {
                csvCamValues.push_back(stof(token));
            }
            line.erase(0, pos + delimeter.length());
        }
        lineNum++;
    }

    // cout << "cam: " << this->camNum << endl;
    // cout << "n params read from csv: " << csvCamValues.size() << endl;
    std::vector<float> k;    // 9
    std::vector<float> r_m;  // 9
    std::vector<float> t;    // 3
    std::vector<float> d;    // 4

    for (int i=0; i<9; i++)
    {
        k.push_back(csvCamValues[i]);
    }
    for (int i=9; i<18; i++)
    {
        r_m.push_back(csvCamValues[i]);
    }
    for (int i=18; i<21; i++)
    {
        t.push_back(csvCamValues[i]);
    }
    for (int i=21; i<25; i++)
    {
        d.push_back(csvCamValues[i]);
    }

    cvp.cam_idx = camNum;

    cvp.K = cv::Mat_<float>(k, true).reshape(0, 3);
    // std::cout << "K = " << std::endl << cv::format(cvp.K, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    cvp.distCoeffs = cv::Mat_<float>(d, true);
    // std::cout << " distCoeffs  = " << std::endl << cv::format(cvp.distCoeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    std::string ty =  type2str( cvp.K.type() );
    // printf("Matrix: %s %dx%d \n", ty.c_str(), cvp.K.cols, cvp.K.rows );

    cvp.r = cv::Mat_<float>(r_m, true).reshape(0, 3);
    // std::cout << "r = " << std::endl << cv::format(cvp.r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    cvp.tvec = cv::Mat_<float>(t, true);
    // std::cout << "tvec = " << std::endl << cv::format(cvp.tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;


    cv::Rodrigues(cvp.r, cvp.rvec);
    // std::cout << "rvec = " << std::endl << cv::format(cvp.rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    cv::sfm::projectionFromKRt(cvp.K, cvp.r, cvp.tvec, cvp.projectionMat);
    // std::cout << "projectionMat = " << std::endl << cv::format(cvp.projectionMat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    return cvp;
}



void yolo_detection(PictureBuffer* display_buffer_cpu, int num_cameras, int* key_num_ptr)
{
    // this detection thread load the model and runs the batch detection 

    cvCamParams CamParamOldFormat[num_cameras];
    std::string cam_file = "/home/red/src/laserCalib/calibres/calibration_2022-11-02_rigspace.csv";

    for(unsigned int i = 0; i < num_cameras; i++){
        CamParamOldFormat[i] = LoadCameraParamsFromCSV(cam_file, i);
    }
 
    // for neural network 
    Detector yolonet = Detector("/home/red/src/Yolo_mark/testrig/yolo-obj.cfg", "/home/red/src/Yolo_mark/testrig/backup/yolo-obj_last.weights", 0, 1);
    auto obj_names = objects_names_from_file("/home/red/src/Yolo_mark/testrig/data/obj.names");

    int detection_frame_id = 0;
    std::vector<std::vector<center_triangulate>> t_res;


    while(*key_num_ptr != 27){
    
        std::vector<std::vector<bbox_t>> result_vec;
        cv::Mat image_src[num_cameras];
        cv::Mat image[num_cameras];

        int frame_id_max = detection_frame_id;    
        // find the biggest frame_id in the buffer, and lock it for detection 
        for (int i = 0; i < num_cameras; i++){
            if (display_buffer_cpu[i].frame_number > frame_id_max){
                frame_id_max = display_buffer_cpu[i].frame_number;                     
            }   
        }

        // if(frame_id_max > detection_frame_id)
        {
            // sync all cameras 
            for (int i = 0; i < num_cameras; i++){
                if(display_buffer_cpu[i].frame_number == frame_id_max){
                    display_buffer_cpu[i].available_to_write = false;
                }
            }


            // detection 
            for(unsigned int i = 0; i < num_cameras; i++){
                image_src[i] = cv::Mat(3208 * 2200 * 3, 1, CV_8U, display_buffer_cpu[i].frame).reshape(3, 2200);
                cv::cvtColor(image_src[i], image[i], cv::COLOR_RGB2BGR);
                result_vec.push_back(yolonet.detect(image[i], 0.85));
            }


            t_res = get_3d_coordinates(result_vec, CamParamOldFormat, num_cameras);


            for (auto &i : t_res[0]){
                std::cout << "x " << i.point_3d.x << ", y " << i.point_3d.y << ", z " << i.point_3d.z << std::endl;
            }
            
            
            for(unsigned int i = 0; i < num_cameras; i++){
                draw_boxes(image_src[i], result_vec[i], obj_names, t_res[i]);

            }


            // unlock 
            for (int i = 0; i < num_cameras; i++){
                display_buffer_cpu[i].available_to_write = true;
            }

        }

        detection_frame_id = frame_id_max;
    }
}