#include "thread.h"
#include "video_capture.h"
#include <opencv2/opencv.hpp>

void aquire_num_frames(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames)
{
    int camera_return {0};

    unsigned int size_of_buffer;
    size_of_buffer = frame_recv->CalculateBufferSize();
    printf("Buffer size (bytes): \t%d\n ", size_of_buffer);

    //aquisition
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;   

    float start_time = tick();
    for(int frame_count=0;frame_count<num_frames;frame_count++)
    {
        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        if(!camera_return)
        {
            //Counting dropped frames through frame_id as redundant check. 
            if(((frame_recv->frame_id) != id_prev+1) && (frame_count != 0)) dropped_frames++;
            else
            {
                frames_recd++;
                // TODO: program what to do with received frame 

            }
        }
        else{dropped_frames++; printf("\nEVT_CameraGetFrame Error = %8.8x!\n", camera_return);}

        //In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if(frame_recv->frame_id == 65535)
            id_prev = 0;
        else
            id_prev = frame_recv->frame_id;

        if(camera_return)
            break; //No requeue reqd

        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.
        if(camera_return) printf("EVT_CameraQueueFrame Error!\n");

        if(frame_count % 100 == 99) {printf("."); fflush(stdout);}    
        if(frame_count % 10000 == 9999) printf("\n");

        if(dropped_frames >= 100) break;
    }
    float end_time = tick();
    float time_diff = end_time - start_time;

    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));

    //Report stats
    printf("\n");
    printf("Images Captured: \t%d\n", frames_recd);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd/time_diff);
}


void aquire_and_display(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, CameraParams camera_params)
{
    int camera_return {0};

    unsigned int size_of_buffer;
    size_of_buffer = frame_recv->CalculateBufferSize();
    printf("Buffer size (bytes): \t%d\n ", size_of_buffer);

    //aquisition
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;   
    int frame_count {0};
    float start_time = tick();
        
    while(true)
    {
        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        if(!camera_return)
        {
            //Counting dropped frames through frame_id as redundant check. 
            if(((frame_recv->frame_id) != id_prev+1) && (frame_count != 0)) dropped_frames++;
            else
            {
                frames_recd++;
                // TODO: program what to do with received frame 
                // using opencv or use opengl for color coversion for display? how does opencv draw windows? 
                cv::Mat frame;
                if(camera_params.pixel_format == "YUV422Packed")
                {
                    cv::Mat frameConvert(camera_params.height, camera_params.width, CV_8UC2, frame_recv->imagePtr);                
                    cv::cvtColor(frameConvert, frame, cv::COLOR_YUV2BGR_Y422);
                }
                else if (camera_params.pixel_format == "BayerRG8")
                {
                    cv::Mat frameConvert(camera_params.height, camera_params.width, CV_8UC1, frame_recv->imagePtr);                
                    cv::cvtColor(frameConvert, frame, cv::COLOR_BayerBG2BGR);
                }
                cv::resize(frame, frame, cv::Size(1604, 1100), 0, 0, cv::INTER_LINEAR);
                cv::imshow("camera", frame);
            }
        }
        else{dropped_frames++; printf("\nEVT_CameraGetFrame Error = %8.8x!\n", camera_return);}

        //In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if(frame_recv->frame_id == 65535)
            id_prev = 0;
        else
            id_prev = frame_recv->frame_id;

        if(camera_return)
            break; //No requeue reqd

        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.
        if(camera_return) printf("EVT_CameraQueueFrame Error!\n");

        if(frame_count % 100 == 99) {printf("."); fflush(stdout);}    
        if(frame_count % 10000 == 9999) printf("\n");

        if(dropped_frames >= 100) break;
        frame_count++;

        char c = (char)cv::waitKey(1);
        if (c==27)              
            break; // ESC for quiting
    }

    float end_time = tick();
    float time_diff = end_time - start_time;
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));

    //Report stats
    printf("\n");
    printf("Images Captured: \t%d\n", frames_recd);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd/time_diff);
}


void aquire_and_encode_ffmpeg(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, CameraParams camera_params)
{
    // encoding using ffmpeg
    FILE *encoder_pipe;
    stringstream str_stream;

    // ffmpeg encode
    string pix_fmt;
    unsigned int size_of_buffer;

    // color conversion in ffmpeg 
    if (camera_params.pixel_format == "YUV422Packed")
    {
        pix_fmt = "uyvy422"; 
        size_of_buffer = camera_params.height * camera_params.width * 2;
    }

    else if (camera_params.pixel_format  == "BayerRG8") 
    {
        pix_fmt = "bayer_rggb8"; // ffmpeg debayer is pretty slow
        size_of_buffer = camera_params.height * camera_params.width;
    }
    printf("Buffer size (bytes): \t%d\n ", size_of_buffer);

    string input_string = "/usr/local/bin/ffmpeg -y -f rawvideo -pix_fmt " + pix_fmt + " -video_size " + to_string(camera_params.width) + "x" + to_string(camera_params.height) + " -framerate " + to_string(camera_params.frame_rate) + " -i - ";
    string output_string = " -c:v h264_nvenc -r " + to_string(camera_params.frame_rate) + " video_pipe_ffmpeg.mp4";
    str_stream << (input_string + output_string);
    //cout  << "Debug: %s", str_stream.str().c_str();
    
    if ( !(encoder_pipe = popen(str_stream.str().c_str(), "w")) ) {
        printf("popen error");
        // think about more general error handling with camera
        exit(1);
    }

    int camera_return {0};

    //aquisition
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;   

    float start_time = tick();

    for(int frame_count=0;frame_count<num_frames;frame_count++)
    {
        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        if(!camera_return)
        {
            //Counting dropped frames through frame_id as redundant check. 
            if(((frame_recv->frame_id) != id_prev+1) && (frame_count != 0)) dropped_frames++;
            else
            {
                frames_recd++;           
                fwrite(frame_recv->imagePtr, 1, size_of_buffer, encoder_pipe);
            }
        }
        else{dropped_frames++; printf("\nEVT_CameraGetFrame Error = %8.8x!\n", camera_return);}

        //In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if(frame_recv->frame_id == 65535)
            id_prev = 0;
        else
            id_prev = frame_recv->frame_id;

        if(camera_return)
            break; //No requeue reqd

        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.
        if(camera_return) printf("EVT_CameraQueueFrame Error!\n");
        if(dropped_frames >= 100) break;
    }

    float end_time = tick();
    float time_diff = end_time - start_time;

    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));
    
    fflush(encoder_pipe);
    fclose(encoder_pipe);

    //Report stats
    printf("\n");
    printf("Images Captured: \t%d\n", frames_recd);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd/time_diff);
}



void aquire_and_encode_gstreamer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, CameraParams camera_params)
{
    // TODO: david's solution, the video container don't scroll; need fix
    cv::VideoWriter writer;
    string gstreamer_option = "appsrc ! videoconvert n-threads=8 ! nvh264enc ! filesink location=video_opencv_gstreamer.mp4";
    if (!(writer.open(gstreamer_option, cv::CAP_GSTREAMER, 0, camera_params.frame_rate, cv::Size(camera_params.width, camera_params.height))))
    {
        cout << "Video writer didn't open. \n";
        exit(1);
    }

    int camera_return {0};
    //aquisition
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;   

    unsigned int size_of_buffer;
    size_of_buffer = frame_recv->CalculateBufferSize();
    
    float start_time = tick();
    
    for(int frame_count=0;frame_count<num_frames;frame_count++)
    {
        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        if(!camera_return)
        {
            //Counting dropped frames through frame_id as redundant check. 
            if(((frame_recv->frame_id) != id_prev+1) && (frame_count != 0)) dropped_frames++;
            else
            {
                frames_recd++;
                cv::Mat frame;
                if(camera_params.pixel_format == "YUV422Packed")
                {
                    cv::Mat frameConvert(camera_params.height, camera_params.width, CV_8UC2, frame_recv->imagePtr);                
                    cv::cvtColor(frameConvert, frame, cv::COLOR_YUV2BGR_Y422);
                }
                else if (camera_params.pixel_format == "BayerRG8")
                {
                    cv::Mat frameConvert(camera_params.height, camera_params.width, CV_8UC1, frame_recv->imagePtr);                
                    cv::cvtColor(frameConvert, frame, cv::COLOR_BayerBG2BGR);
                }
                writer.write(frame);                
            }
        }
        else{dropped_frames++; printf("\nEVT_CameraGetFrame Error = %8.8x!\n", camera_return);}

        //In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if(frame_recv->frame_id == 65535)
            id_prev = 0;
        else
            id_prev = frame_recv->frame_id;

        if(camera_return)
            break; //No requeue reqd

        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.
        if(camera_return) printf("EVT_CameraQueueFrame Error!\n");
        if(dropped_frames >= 100) break;
    }

    float end_time = tick();
    float time_diff = end_time - start_time;

    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));
    
    //Report stats
    printf("\n");
    printf("Images Captured: \t%d\n", frames_recd);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd/time_diff);
}