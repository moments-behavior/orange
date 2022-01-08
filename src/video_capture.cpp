#include "thread.h"
#include "video_capture.h"
#include <opencv2/opencv.hpp>

void aquire_num_frames(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, bool save_bmp)
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
                if (save_bmp)
                {
                    string frame_name {};
                    frame_name = "../frames/frame_%5d_.bmp" + frame_recv->frame_id;
                    EVT_FrameSave(frame_recv, frame_name.c_str(), EVT_FILETYPE_BMP, EVT_ALIGN_NONE); 
                }
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

    string output_string = " -c:v h264_nvenc -r " + to_string(camera_params.frame_rate) + " -preset p2 -f mp4 video_pipe_ffmpeg.mp4";

    //cout  << "Debug: %s", str_stream.str().c_str();

    // color conversion in ffmpeg 
    if (camera_params.pixel_format == "YUV422Packed")
    {
        // note: if it is uyvy422, it donesn't need color conversion, but 420p results in main profile, whre uyvy422 results in high profile, might be better for storage 
        pix_fmt = "uyvy422"; 
        size_of_buffer = camera_params.height * camera_params.width * 2;
        printf("Buffer size (bytes): \t%d\n ", size_of_buffer);
        string input_string = "/usr/local/bin/ffmpeg -y -f rawvideo -pixel_format " + pix_fmt + " -video_size " + to_string(camera_params.width) + "x" + to_string(camera_params.height) + " -framerate " + to_string(camera_params.frame_rate) + " -i - ";
        string color_conv_string = "-f rawvideo -pix_fmt yuv420p -video_size " + to_string(camera_params.width) + "x" + to_string(camera_params.height) + " -framerate " + to_string(camera_params.frame_rate);
        str_stream << (input_string + output_string);

    }
    else if (camera_params.pixel_format  == "BayerRG8") 
    {
        pix_fmt = "bayer_rggb8"; 
        size_of_buffer = camera_params.height * camera_params.width;
        printf("Buffer size (bytes): \t%d\n ", size_of_buffer);
        string input_string = "/usr/local/bin/ffmpeg -y -f rawvideo -pixel_format " + pix_fmt + " -video_size " + to_string(camera_params.width) + "x" + to_string(camera_params.height) + " -framerate " + to_string(camera_params.frame_rate) + " -i - ";
        string color_conv_string = "-f rawvideo -pix_fmt yuv420p -video_size " + to_string(camera_params.width) + "x" + to_string(camera_params.height) + " -framerate " + to_string(camera_params.frame_rate);
        str_stream << (input_string + color_conv_string + output_string);
    }
    else{
        printf("Color format not supported! \n");
        throw(EXIT_FAILURE);
    }



    if ( !(encoder_pipe = popen(str_stream.str().c_str(), "w")) ) {
        printf("popen error");
        // think about more general error handling with camera
        throw(EXIT_FAILURE);
    }

    int camera_return {0};

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
        if(dropped_frames >= 100) 
        {
            printf("More than 100 dropped frames. Exit.");
            break;
        }
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
    //cv writer need 3 channels, interesting 
    cv::VideoWriter writer;
    string gstreamer_option = "appsrc ! videoconvert n-threads=8 ! nvh264enc ! filesink location=video_opencv_gstreamer.mp4";
    if (!(writer.open(gstreamer_option, cv::CAP_GSTREAMER, 0, camera_params.frame_rate, cv::Size(camera_params.width, camera_params.height))))
    {
        cout << "Video writer didn't open. \n";
        throw(EXIT_FAILURE);
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
        if(dropped_frames >= 100) break;        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.

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