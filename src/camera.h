#ifndef ORANGE_CAMERA
#define ORANGE_CAMERA

#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <EvtParamAttribute.h>
#include <gigevisiondeviceinfo.h>


struct CameraParams{
    unsigned int frame_rate;
    unsigned int gain;
    unsigned int exposure;
    string pixel_format;
    string color_temp;
}; 

// struct Camera{
//     Emergent::CEmergentCamera* emergent_cam;
//     CameraParams params;
//     Emergent::CEmergentFrame* evtFrame; 
//     Emergent::CEmergentFrame* evtFrameRecv; 
// };

CameraParams create_camera_params(unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp);
int get_number_cameras(int max_cameras, GigEVisionDeviceInfo* deviceInfo);
void configure_factory_defaults(Emergent::CEmergentCamera* camera);
void close_camera(Emergent::CEmergentCamera* camera);
int set_camera_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params);
string evtGetErrorString(EVT_ERROR error);


#ifndef checkCameraErrors
#define checkCameraErrors(err) __checkCameraErrors(err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
inline void __checkCameraErrors(EVT_ERROR err, const char *file, const int line) {
  if (EVT_SUCCESS != err) {
    string error_string;
    error_string = evtGetErrorString(err);
    const char*  errorStr = error_string.c_str();
    fprintf(stderr,
            "checkCameraErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif


#endif
