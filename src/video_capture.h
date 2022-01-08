#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE

#include <iomanip>
#include "camera.h"

void aquire_num_frames(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, bool save_bmp);
void aquire_and_encode_ffmpeg(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, CameraParams camera_params);
void aquire_and_display(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, CameraParams camera_params);
void aquire_and_encode_gstreamer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, CameraParams camera_params);

#endif 