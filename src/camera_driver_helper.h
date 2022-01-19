#ifndef ORANGE_CAMERA_DRIVER_HELPER
#define ORANGE_CAMERA_DRIVER_HELPER

#ifndef  EMERGENT_SDK
#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <EvtParamAttribute.h>
#include <gigevisiondeviceinfo.h>
#endif

string get_evt_error_string(EVT_ERROR error);

#define check_camera_errors(err) __check_camera_errors(err, __FILE__, __LINE__)
inline void __check_camera_errors(EVT_ERROR err, const char *file, const int line) {
  if (EVT_SUCCESS != err) {
    string error_string;
    error_string = get_evt_error_string(err);
    const char*  errorStr = error_string.c_str();
    fprintf(stderr,
            "checkCameraErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

void print_camera_device_struct(GigEVisionDeviceInfo* device_info, int camera_idx);

#endif
