#include "thread.h"
#include "video_capture.h"

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



void aquire_and_encode_ffmpeg(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, CameraParams camera_params, FILE* encoder_stream)
{
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
                // write to pipe
                fwrite(frame_recv->imagePtr, 1, size_of_buffer, encoder_stream);
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