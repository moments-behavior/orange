#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "opengldisplay.h"
#include <cuda_runtime_api.h>


int trigger_count = 0;
int trigger_count_threshold = 10;


COpenGLDisplay::COpenGLDisplay( const char *name, 
                                CameraParams *camera_params, 
                                CameraEachSelect *camera_select, 
                                unsigned char *display_buffer, 
                                INDIGOSignalBuilder* indigo_signal_builder
                                )
    : CThreadWorker(name), camera_params(camera_params), camera_select(camera_select), display_buffer(display_buffer), indigo_signal_builder(indigo_signal_builder)
{
    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = WORK_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++)
    {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
}

COpenGLDisplay::~COpenGLDisplay()
{
}

void COpenGLDisplay::ThreadRunning()
{
    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);
    initialize_cpu_frame(&frame_cpu, camera_params);

    ck(cudaMalloc((void **)&d_convert, camera_params->width * camera_params->height * 3)); 
    
    unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // box
    if (camera_select->yolo) {
        printf("YOLO initialization...\n");

        const std::string engine_file_path = camera_select->yolo_model;
        yolov8 = new YOLOv8(engine_file_path, camera_params->width, camera_params->height);
        yolov8->make_pipe(true);

        cudaMalloc((void **)&d_points, sizeof(float) * 16);
        cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
        CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice));
    }

    std::vector<Object> objs;
    std::vector<Object> objs_last_frame; // think about better way that scales with frame
 
    while(IsMachineOn())
    {
        void* f = GetObjectFromQueueIn();
        if(f) {
            WORKER_ENTRY entry = *(WORKER_ENTRY*)f;
            PutObjectToQueueOut(f);
            
            // copy frame from cpu to gpu
            ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, entry.imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));

            if (camera_params->color){
                debayer_frame_gpu(camera_params, &frame_original, &debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_original, &debayer);
            }

            if (camera_select->yolo) {
                rgba2rgb_convert(d_convert, debayer.d_debayer, camera_params->width, camera_params->height, 0);
                yolov8->preprocess_gpu(d_convert);
                yolov8->infer();
                yolov8->postprocess(objs);

                // place holder for pose message
                float x_center, x_mouse,x_ball,w_mouse = -1.0;
                float y_center, y_mouse,y_ball,h_mouse = -1.0;
                float width = -1.0;
                float height = -1.0;
                float theta = -999.0;
                float prob = -1;
                float label = -1;



                // //  messages and their defaults
                
                // uint8_t *buffer_pointer = indigo_signal_builder->builder->GetBufferPointer();
                // auto pose_msg_mutable = ObjPose::GetMutableobj_pose_msg(buffer_pointer);                
                
                // pose_msg_mutable->mutable_ball()->mutate_x(float(x_center));
                // pose_msg_mutable->mutable_ball()->mutate_y(float(y_center));
                // pose_msg_mutable->mutable_ball()->mutate_theta(float(theta));
                // pose_msg_mutable->mutable_ball()->mutate_prob(float(prob));
                // pose_msg_mutable->mutable_ball()->mutate_label(float(label));

                // pose_msg_mutable->mutable_mouse()->mutate_x(float(x_center));
                // pose_msg_mutable->mutable_mouse()->mutate_y(float(y_center));
                // pose_msg_mutable->mutable_mouse()->mutate_theta(float(theta));
                // pose_msg_mutable->mutable_mouse()->mutate_prob(float(prob));
                // pose_msg_mutable->mutable_mouse()->mutate_label(float(label));
                        

                for (auto obj : objs ){
                    x_center = obj.rect.x + obj.rect.width/2;
                    y_center = obj.rect.y + obj.rect.height/2;
                    prob = obj.prob;
                    label = obj.label;
                    
                    if(obj.label==0) {//mouse
                        std::cout <<"ball at (x,y) = (" ;
                    }
                    else { //ball
                        std::cout <<"ball at (x,y) = (" ;
                    }

                    std::cout <<x_center << "," << y_center << ")" << std::endl; 
                }

                // extract object IDs
                int id_mouse = -1;
                int id_ball = -1;
                for (int ii =0 ; ii<objs.size(); ii++) {

                    

                    if(objs[ii].prob >0.65)  { // only check for confident detections
                        switch(objs[ii].label) {
                            case 0: 
                                //bbox level checks for mouse
                                if( (max(objs[ii].rect.width, objs[ii].rect.height) <= 300 ) &&
                                    (min(objs[ii].rect.width, objs[ii].rect.height) >= 50  ) )  {
                                        
                                        x_mouse =  objs[ii].rect.x + objs[ii].rect.width/2;;
                                        y_mouse =  objs[ii].rect.y + objs[ii].rect.height/2;
                                        h_mouse = objs[ii].rect.height;
                                        w_mouse = objs[ii].rect.width;
                                        id_mouse=0; 

                                    }
                                    
                                break;
                            case 1:
                                if( (max(objs[ii].rect.width, objs[ii].rect.height) <= 300 ) &&
                                    (min(objs[ii].rect.width, objs[ii].rect.height) >= 50  ) )  {

                                        x_ball =  objs[ii].rect.x + objs[ii].rect.width/2;;
                                        y_ball =  objs[ii].rect.y + objs[ii].rect.height/2;
                                        id_ball=1;
                                }

                                break;
                        }

                    }
                }

                // check for mouse getting closer to ball
                if(id_mouse>-1 && id_ball > -1){
                    float dx = std::min( abs(x_mouse - x_ball), 
                                     abs(x_mouse + w_mouse - x_ball));

                    float dy = std::min( abs(y_mouse - y_ball), 
                                        abs(y_mouse + h_mouse - y_ball));
                                    
                    float d_ball_center_to_mouse_corner  = pow(dx*dx + dy*dy,0.5);
                    float r_cutoff =  1.414*1.25*100; // sqrt(2)*scale*ball_size_px

                    if(d_ball_center_to_mouse_corner - r_cutoff*r_cutoff <=0)
                    {   

                        trigger_count++;
                        if(trigger_count==trigger_count_threshold)
                        {
                            std::cout << "trigger reward" << std::endl;                        
                            if (indigo_signal_builder->indigo_connection != NULL) {
                                send_indigo_message(indigo_signal_builder->server, indigo_signal_builder->builder, indigo_signal_builder->indigo_connection, FetchGame::SignalType_INDIGO_TRIAL_SUCCESS);
                            }
                            trigger_count = 0;
                        }
                    }
                    else
                    {
                        trigger_count = max(0,trigger_count--);
                    }

                }
                                                                    
                // draw objects
                std::cout << objs.size() << " objects detected; plotted " ;
                if (objs.size()>0)
                {

                    for (int ii = 0; ii< objs.size(); ii++){
                        
                        yolov8->copy_keypoints_gpu(d_points, objs[ii]);
                        if(objs[ii].label==0){
                            gpu_draw_rat_pose(debayer.d_debayer, camera_params->width, camera_params->height, d_points, d_skeleton, yolov8->stream);

                        }
                        else {
                            gpu_draw_ring(debayer.d_debayer, camera_params->width, camera_params->height, d_points, 1.25, yolov8->stream);

                        }
                        std::cout<< " obj " << ii;
                    }
                }
                std::cout << " " <<std::endl;                        
                
            }

            // probably reduandant copy
            ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));

            if (camera_select->frame_save_state==State_Write_New_Frame) {
                // yolo code goes here
                rgba2bgr_convert(d_convert, debayer.d_debayer, camera_params->width, camera_params->height, 0);
                
                // copy frame back to cpu, and then save
                cudaMemcpy2D(frame_cpu.frame, camera_params->width*3, d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost);
                cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, frame_cpu.frame).reshape(3, camera_params->height);
                
                std::string image_name = camera_select->picture_save_folder + "/" + camera_params->camera_serial + "_" + camera_select->frame_save_name + ".tiff";
                cv::imwrite(image_name, view);
                camera_select->pictures_counter++;
                camera_select->frame_save_state = State_Frame_Idle;
            }          

        }
        usleep(16000); // sleep for 16ms 
    }
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (camera_select->yolo) {
        delete yolov8;
    }
}


bool COpenGLDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id)
{
    WORKER_ENTRY *entriesOut[WORK_ENTRIES_MAX]; // entris got out from saver thread, their frames should be returned to driver queue.
    int entriesOutCount = WORK_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount)
    { // return the frames to driver, and put entries back to frameSaveEntriesFreeQueue
        // printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__, entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++)
        {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount)
    {
        // printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__, workerEntriesFreeQueueCount);
        WORKER_ENTRY *entry = workerEntriesFreeQueue[workerEntriesFreeQueueCount - 1];
        workerEntriesFreeQueueCount--;
        entry->imagePtr = imagePtr;
        entry->bufferSize = bufferSize;
        entry->width = width;
        entry->height = height;
        entry->pixelFormat = pixelFormat;
        entry->timestamp = timestamp;
        entry->frame_id = frame_id;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}