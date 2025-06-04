#ifndef ORANGE_ACQUIRE_FRAMES
#define ORANGE_ACQUIRE_FRAMES
#include "video_capture.h"
#include "yolo_worker.h"
void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params, CameraEachSelect* camera_select, CameraControl* camera_control, unsigned char *display_buffer, std::string encoder_setup, std::string folder_name, PTPParams* ptp_params, INDIGOSignalBuilder* indigo_signal_builder, YOLOv8Worker* yolo_worker_for_this_camera);
#endif