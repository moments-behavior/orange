#ifndef ORANGE_ACQUIRE_FRAMES
#define ORANGE_ACQUIRE_FRAMES
#include "video_capture.h"
#include "realtime_tool.h"
#include "detection3D.h"
void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params, CameraEachSelect* camera_select, CameraControl *camera_control, unsigned char *display_buffer, std::string encoder_setup, std::string folder_name, PTPParams *ptp_params, INDIGOSignalBuilder* indigo_signal_builder, DetectionData* detection_data, SyncDetection* sync_detection);
#endif