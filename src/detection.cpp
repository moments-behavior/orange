#include "detection.h"


std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

struct cvCamParams {
    int cam_idx;
    cv::Mat K;
    cv::Mat distCoeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projectionMat;
};

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

    std::vector<std::vector<bbox_t>> result_vec;
    int detection_frame_id = 0;
    
    while(*key_num_ptr != 27){
        
        int frame_id_max = detection_frame_id;    
        // find the biggest frame_id in the buffer, and lock it for detection 
        for (int i = 0; i < num_cameras; i++){
            if (display_buffer_cpu[i].frame_number > frame_id_max){
                frame_id_max = display_buffer_cpu[i].frame_number;                     
            }   
        }

        if(frame_id_max > detection_frame_id)
        {
            // sync all cameras 
            for (int i = 0; i < num_cameras; i++){
                if(display_buffer_cpu[i].frame_number == frame_id_max){
                    display_buffer_cpu[i].available_to_write = false;
                }
            }


            // detection 
            for(unsigned int i = 0; i < num_cameras; i++){
                cv::Mat image = cv::Mat(3208 * 2200 * 4, 1, CV_8U, display_buffer_cpu[i].frame).reshape(4, 2200);
                cv::cvtColor(image, image , cv::COLOR_RGBA2BGR);
                result_vec.push_back(yolonet.detect(image, 0.85));
            }


            std::vector<std::vector<center_triangulate>> t_res = get_3d_coordinates(result_vec, CamParamOldFormat);


            for(unsigned int cam = 0; cam < num_cameras; cam++){

                // // for each marker, draw info and its boundaries in the image
                // for (unsigned int i = 0; i < Markers[cam].size(); i++)
                // {
                //     cout << Markers[cam][i] << endl;
                //     Markers[cam][i].draw(InImage[cam], cv::Scalar(0, 0, 255), 2);
                // }



                // for yolo detection;
                draw_boxes(display_buffer_cpu[i].frame, result_vec[cam], obj_names, t_res[cam]);
            }



            // unlock 
            for (int i = 0; i < num_cameras; i++){
                display_buffer_cpu[i].available_to_write = true;
            }

        }
        detection_frame_id = frame_id_max;
    }
}