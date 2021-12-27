#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE

#include <iomanip>
#include "camera.h"

void aquire_num_frames(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames);
void aquire_and_encode_ffmpeg(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames, CameraParams camera_params, FILE* encoder_pipe);


#endif 