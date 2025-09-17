#ifndef ORANGE_DETECTION3D
#define ORANGE_DETECTION3D
#include "video_capture.h"

void detection3d_proc(CameraControl *camera_control,
                      CameraEachSelect *cameras_select, int num_cameras);

// NEW: Jarvis 3D pose processing
void jarvis_3d_pose_proc(CameraControl *camera_control,
                         CameraEachSelect *cameras_select, int num_cameras);
#endif
