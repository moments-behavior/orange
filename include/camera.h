#ifndef ORANGE_CAMERA
#define ORANGE_CAMERA

#include <emergentcameradef.h>
#include <emergentgigevisiondef.h>
#include <EvtParamAttribute.h>
#include <unistd.h>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>

std::string get_evt_error_string(EVT_ERROR error);

#define check_camera_errors(err, camera_serial) __check_camera_errors(err, camera_serial, __FILE__, __LINE__)

inline void __check_camera_errors(EVT_ERROR err, const char *camera_serial, const char *file, const int line) {
  if (EVT_SUCCESS != err) {
    std::string error_string;
    error_string = get_evt_error_string(err);
    const char*  errorStr = error_string.c_str();
    fprintf(stderr,
            "%s checkCameraErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            camera_serial, err, errorStr, file, line);
    throw(EXIT_FAILURE);
  }
}

struct CameraEmergent{
    Emergent::CEmergentCamera camera;
    Emergent::CEmergentFrame* evt_frame;
    Emergent::CEmergentFrame frame_recv;
    Emergent::CEmergentFrame frame_reorder;
};

#endif