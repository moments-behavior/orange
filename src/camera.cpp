#include "camera.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

std::string get_evt_error_string(EVT_ERROR error) {
    std::string error_string;
    switch (error) {
    case EVT_ENOENT:
        error_string = "No such file or directory.";
        break;
    case EVT_ERROR_SRCH:
        error_string = "No such process.";
        break;
    case EVT_ERROR_IO:
        error_string = "I/O error";
        break;
    case EVT_ERROR_ECHILD:
        error_string = "Child process or thread create error.";
        break;
    case EVT_ERROR_AGAIN:
        error_string = "Try again";
        break;
    case EVT_ERROR_NOMEM:
        error_string = "Out of memory.";
        break;
    case EVT_ERROR_ACCES:
        error_string = "No access or permission.";
        break;
    case EVT_ERROR_ENODEV:
        error_string = "No such device.";
        break;
    case EVT_ERROR_INVAL:
        error_string = "Invalid argument.";
        break;
    case EVT_ERROR_NOT_SUPPORTED:
        error_string = "Not supported.";
        break;
    case EVT_ERROR_DEVICE_CONNECTED_ALRD:
        error_string = "Camera has been opened.";
        break;
    case EVT_ERROR_DEVICE_NOT_CONNECTED:
        error_string = "Camera has not been opened.";
        break;
    case EVT_ERROR_DEVICE_LOST_CONNECTION:
        error_string = "Camera lost connection due to disconnected, powered "
                       "off, crashing etc.";
        break;
    case EVT_ERROR_GENICAM_ERROR:
        error_string = "Generic GeniCam error from GeniCam lib.";
        break;
    case EVT_ERROR_GENICAM_NOT_MATCH:
        error_string = "Parameter not matched.";
        break;
    case EVT_ERROR_GENICAM_OUT_OF_RANGE:
        error_string = "Parameter out of range.";
        break;
    case EVT_ERROR_SOCK:
        error_string = "Socket operation failed.";
        break;
    case EVT_ERROR_GVCP_ACK:
        error_string = "GVCP ACK error.";
        break;
    case EVT_ERROR_GVSP_DATA_CORRUPT:
        error_string = "Gvsp stream data corrupted, would cause block dropped.";
        break;
    case EVT_ERROR_NIC_LIB_INIT:
        error_string = "Fail to initialize NIC's SDK library.";
        break;
    case EVT_ERROR_OS_OBTAIN_ADAPTER:
        error_string = "Failed to get host adapter info.";
        break;
    case EVT_ERROR_SDK:
        error_string = "SDK error, should not occur. Can be removed if sdk is "
                       "proved to be correct.";
        break;
    case EVT_GENERAL_ERROR:
        error_string = "General error.";
        break;
    }
    return error_string;
}

void print_camera_device_struct(GigEVisionDeviceInfo *device_info,
                                int camera_idx) {
    std::cout << "Camera: " << camera_idx << std::endl;
    std::cout << "userDefinedName: " << device_info[camera_idx].userDefinedName
              << std::endl;
    std::cout << "macAddress: " << device_info[camera_idx].macAddress
              << std::endl;
    std::cout << "deviceMode: " << device_info[camera_idx].deviceMode
              << std::endl;
    std::cout << "serialNumber: " << device_info[camera_idx].serialNumber
              << std::endl;
    std::cout << "macAddress: " << device_info[camera_idx].macAddress
              << std::endl;
    std::cout << "currentIp: " << device_info[camera_idx].currentIp
              << std::endl;
    std::cout << "currentSubnetMask: "
              << device_info[camera_idx].currentSubnetMask << std::endl;
    std::cout << "defaultGateway: " << device_info[camera_idx].defaultGateway
              << std::endl;
    std::cout << "nic.ip4Address: " << device_info[camera_idx].nic.ip4Address
              << std::endl;
}

// A function to reset to factory defaults for running eSDK examples
//  TODO: many thing doesn't work with this emergent native code
void configure_factory_defaults(Emergent::CEmergentCamera *camera,
                                CameraParams *camera_params) {
    unsigned int width_max, height_max, param_val_max;
    // const unsigned long enumBufferSize = 1000;
    // unsigned long enumBufferSizeReturn = 0;
    // char enumBuffer[enumBufferSize];
    // char* next_token;
    // char* enumMember = strtok_s(enumBuffer, ",", &next_token);

    // Order is important as param max/mins get updated.
    // check_camera_errors(Emergent::EVT_CameraGetEnumParamRange(camera,
    // "PixelFormat", enumBuffer, enumBufferSize, &enumBufferSizeReturn));
    // check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera,
    // "PixelFormat", enumMember));
    //  check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera,
    //  "FrameRate", 30));

    check_camera_errors(
        Emergent::EVT_CameraSetUInt32Param(camera, "OffsetX", 0),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetUInt32Param(camera, "OffsetY", 0),
        camera_params->camera_serial.c_str());

    check_camera_errors(
        Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max),
        camera_params->camera_serial.c_str());
    // check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "Width",
    // width_max));

    check_camera_errors(
        Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max),
        camera_params->camera_serial.c_str());
    // check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "Height",
    // height_max));

    check_camera_errors(Emergent::EVT_CameraSetEnumParam(
                            camera, "AcquisitionMode", "Continuous"),
                        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1),
        camera_params->camera_serial.c_str());
    check_camera_errors(Emergent::EVT_CameraSetEnumParam(
                            camera, "TriggerSelector", "AcquisitionStart"),
                        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetEnumParam(camera, "TriggerMode", "Off"),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetEnumParam(camera, "TriggerSource", "Software"),
        camera_params->camera_serial.c_str());

    // check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera,
    // "BufferMode", "Off"), camera_params->camera_serial.c_str());
    // check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera,
    // "BufferNum", 0), camera_params->camera_serial.c_str());

    check_camera_errors(Emergent::EVT_CameraGetUInt32ParamMax(
                            camera, "GevSCPSPacketSize", &param_val_max),
                        camera_params->camera_serial.c_str());
    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(
                            camera, "GevSCPSPacketSize", param_val_max),
                        camera_params->camera_serial.c_str());

    // check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "Gain",
    // 1000)); check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera,
    // "Offset", 0));

    check_camera_errors(
        Emergent::EVT_CameraSetBoolParam(camera, "LUTEnable", false),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetBoolParam(camera, "AutoGain", false),
        camera_params->camera_serial.c_str());
}

void get_senstemp_range(Emergent::CEmergentCamera *camera,
                        CameraParams *camera_params) {
    EVT_CameraGetInt32ParamMax(camera, "SensTemp",
                               &camera_params->sens_temp_max);
    EVT_CameraGetInt32ParamMin(camera, "SensTemp",
                               &camera_params->sens_temp_min);
}

void get_senstemp_value(Emergent::CEmergentCamera *camera,
                        CameraParams *camera_params) {
    int return_value =
        EVT_CameraGetInt32Param(camera, "SensTemp", &camera_params->sens_temp);
    if (return_value != 0) {
        printf("get_senstemp_value: Error\n");
    }
}

void update_gain_value(Emergent::CEmergentCamera *camera, int gain_val,
                       CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Gain", &camera_params->gain_max);
    EVT_CameraGetUInt32ParamMin(camera, "Gain", &camera_params->gain_min);
    EVT_CameraGetUInt32ParamInc(camera, "Gain", &camera_params->gain_inc);
    if (gain_val >= camera_params->gain_min &&
        gain_val <= camera_params->gain_max) {
        EVT_CameraSetUInt32Param(camera, "Gain", gain_val);
        camera_params->gain = gain_val;
    }
}

void update_color_temperature(Emergent::CEmergentCamera *camera,
                              std::string color_string,
                              CameraParams *camera_params) {
    const char *color_temp = color_string.c_str();
    check_camera_errors(EVT_CameraSetEnumParam(camera, "ColorTemp", color_temp),
                        camera_params->camera_serial.c_str());
    camera_params->color_temp = color_string;
}

void update_focus_value(Emergent::CEmergentCamera *camera, int focus_value,
                        CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Focus", &camera_params->focus_max);
    EVT_CameraGetUInt32ParamMin(camera, "Focus", &camera_params->focus_min);
    EVT_CameraGetUInt32ParamInc(camera, "Focus", &camera_params->focus_inc);
    if (focus_value >= camera_params->focus_min &&
        focus_value <= camera_params->focus_max) {
        EVT_CameraSetUInt32Param(camera, "Focus", focus_value);
        camera_params->focus = focus_value;
    }
}

void update_iris_value(Emergent::CEmergentCamera *camera, int iris_value,
                       CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Iris", &camera_params->iris_max);
    EVT_CameraGetUInt32ParamMin(camera, "Iris", &camera_params->iris_min);
    EVT_CameraGetUInt32ParamInc(camera, "Iris", &camera_params->iris_inc);
    if (iris_value >= camera_params->iris_min &&
        iris_value <= camera_params->iris_max) {
        EVT_CameraSetUInt32Param(camera, "Iris", iris_value);
        camera_params->iris = iris_value;
    }
}

void update_width_value(Emergent::CEmergentCamera *camera, int width_val,
                        CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Width", &camera_params->width_max);
    EVT_CameraGetUInt32ParamMin(camera, "Width", &camera_params->width_min);
    EVT_CameraGetUInt32ParamInc(camera, "Width", &camera_params->width_inc);
    if (width_val >= camera_params->width_min &&
        width_val <= camera_params->width_max) {
        EVT_CameraSetUInt32Param(camera, "Width", width_val);
        camera_params->width = width_val;
    }
}

void update_height_value(Emergent::CEmergentCamera *camera, int height_val,
                         CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Height", &camera_params->height_max);
    EVT_CameraGetUInt32ParamMin(camera, "Height", &camera_params->height_min);
    EVT_CameraGetUInt32ParamInc(camera, "Height", &camera_params->height_inc);
    if (height_val >= camera_params->height_min &&
        height_val <= camera_params->height_max) {
        EVT_CameraSetUInt32Param(camera, "Height", height_val);
        camera_params->height = height_val;
    }
}

void update_exposure_value(Emergent::CEmergentCamera *camera, int exposure_val,
                           CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Exposure",
                                &camera_params->exposure_max);
    EVT_CameraGetUInt32ParamMin(camera, "Exposure",
                                &camera_params->exposure_min);
    EVT_CameraGetUInt32ParamInc(camera, "Exposure",
                                &camera_params->exposure_inc);

    if (exposure_val >= camera_params->exposure_min &&
        exposure_val <= camera_params->exposure_max) {
        EVT_CameraSetUInt32Param(camera, "Exposure", exposure_val);
        camera_params->exposure = exposure_val;
    }
}

void update_exposure_framerate_value(Emergent::CEmergentCamera *camera,
                                     int exposure_val, int *frame_rate_val,
                                     CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "Exposure",
                                &camera_params->exposure_max);
    EVT_CameraGetUInt32ParamMin(camera, "Exposure",
                                &camera_params->exposure_min);
    EVT_CameraGetUInt32ParamInc(camera, "Exposure",
                                &camera_params->exposure_inc);

    if (exposure_val >= camera_params->exposure_min &&
        exposure_val <= camera_params->exposure_max) {
        EVT_CameraSetUInt32Param(camera, "Exposure", exposure_val);
        camera_params->exposure = exposure_val;

        // framerate is correlated with exposure
        EVT_CameraGetUInt32ParamMax(camera, "FrameRate",
                                    &camera_params->frame_rate_max);
        EVT_CameraGetUInt32ParamMin(camera, "FrameRate",
                                    &camera_params->frame_rate_min);
        EVT_CameraGetUInt32ParamInc(camera, "FrameRate",
                                    &camera_params->frame_rate_inc);

        if (*frame_rate_val < camera_params->frame_rate_min) {
            *frame_rate_val = camera_params->frame_rate_min;
        } else if (*frame_rate_val > camera_params->frame_rate_max) {
            *frame_rate_val = camera_params->frame_rate_max;
        }

        EVT_CameraSetUInt32Param(camera, "FrameRate", *frame_rate_val);
        camera_params->frame_rate = *frame_rate_val;
    }
}

void update_frame_rate_value(Emergent::CEmergentCamera *camera,
                             int frame_rate_val, CameraParams *camera_params) {
    EVT_CameraGetUInt32ParamMax(camera, "FrameRate",
                                &camera_params->frame_rate_max);
    EVT_CameraGetUInt32ParamMin(camera, "FrameRate",
                                &camera_params->frame_rate_min);
    EVT_CameraGetUInt32ParamInc(camera, "FrameRate",
                                &camera_params->frame_rate_inc);
    if (frame_rate_val >= camera_params->frame_rate_min &&
        frame_rate_val <= camera_params->frame_rate_max) {
        EVT_CameraSetUInt32Param(camera, "FrameRate", frame_rate_val);
        camera_params->frame_rate = frame_rate_val;
    }
}

void update_offsetX_value(Emergent::CEmergentCamera *camera, int OFFSET_X_VAL,
                          CameraParams *camera_params) {
    // Set ROI OffsetX. Now that Width changed we need to check new OffsetX
    // limits
    EVT_CameraGetUInt32ParamMax(camera, "OffsetX", &camera_params->offsetx_max);
    printf("OffsetX Max: \t\t%d\n", camera_params->offsetx_max);
    EVT_CameraGetUInt32ParamMin(camera, "OffsetX", &camera_params->offsetx_min);
    printf("OffsetX Min: \t\t%d\n", camera_params->offsetx_min);
    EVT_CameraGetUInt32ParamInc(camera, "OffsetX", &camera_params->offsetx_inc);
    printf("OffsetX Inc: \t\t%d\n", camera_params->offsetx_inc);

    if (OFFSET_X_VAL >= camera_params->offsetx_min &&
        OFFSET_X_VAL <= camera_params->offsetx_max) {
        EVT_CameraSetUInt32Param(camera, "OffsetX", OFFSET_X_VAL);
        camera_params->offsetx = OFFSET_X_VAL;
        printf("OffsetX Set: \t\t%d\n", OFFSET_X_VAL);
    }
}

void update_offsetY_value(Emergent::CEmergentCamera *camera, int OFFSET_Y_VAL,
                          CameraParams *camera_params) {
    // Set ROI OffsetX. Now that Width changed we need to check new OffsetX
    // limits
    EVT_CameraGetUInt32ParamMax(camera, "OffsetY", &camera_params->offsety_max);
    printf("OffsetY Max: \t\t%d\n", camera_params->offsety_max);
    EVT_CameraGetUInt32ParamMin(camera, "OffsetY", &camera_params->offsety_min);
    printf("OffsetY Min: \t\t%d\n", camera_params->offsety_min);
    EVT_CameraGetUInt32ParamInc(camera, "OffsetY", &camera_params->offsety_inc);
    printf("OffsetY Inc: \t\t%d\n", camera_params->offsety_inc);

    if (OFFSET_Y_VAL >= camera_params->offsety_min &&
        OFFSET_Y_VAL <= camera_params->offsety_max) {
        EVT_CameraSetUInt32Param(camera, "OffsetY", OFFSET_Y_VAL);
        camera_params->offsety = OFFSET_Y_VAL;
        printf("OffsetX Set: \t\t%d\n", OFFSET_Y_VAL);
    }
}

void open_camera_with_params(Emergent::CEmergentCamera *camera,
                             GigEVisionDeviceInfo *device_info,
                             CameraParams *camera_params) {
    // TODO: open camera using xml file after explored on camera settings
    // EVT_CameraOpen(&camera, &deviceInfo[camera_index], XML_FILE);

    if (camera_params->gpu_direct) {
        camera->gpuDirectDeviceId = camera_params->gpu_id;
    }

    check_camera_errors(EVT_CameraOpen(camera, device_info),
                        camera_params->camera_serial.c_str());

    configure_factory_defaults(camera, camera_params);

    unsigned int width_max, height_max;
    check_camera_errors(
        Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max),
        camera_params->camera_serial.c_str());
    printf("Resolution: \t\t%d x %d\n", width_max, height_max);

    update_width_value(camera, camera_params->width, camera_params);
    update_height_value(camera, camera_params->height, camera_params);

    update_offsetX_value(camera, 0, camera_params);
    update_offsetY_value(camera, 0, camera_params);

    const char *pixel_format = camera_params->pixel_format.c_str();
    check_camera_errors(
        EVT_CameraSetEnumParam(camera, "PixelFormat", pixel_format),
        camera_params->camera_serial.c_str());
    printf("PixelFormat: \t\t%s\n", pixel_format);

    if (camera_params->color) {
        const char *color_temp = camera_params->color_temp.c_str();
        check_camera_errors(
            EVT_CameraSetEnumParam(camera, "ColorTemp", color_temp),
            camera_params->camera_serial.c_str());
    }

    // check_camera_errors(EVT_CameraSetUInt32Param(camera, "Gain",
    // camera_params.gain));
    update_gain_value(camera, camera_params->gain, camera_params);

    // check_camera_errors(EVT_CameraSetUInt32Param(camera, "Exposure",
    // camera_params->exposure));
    update_exposure_value(camera, camera_params->exposure, camera_params);

    // unsigned int frame_rate_max;
    // check_camera_errors(EVT_CameraGetUInt32ParamMax(camera, "FrameRate",
    // &frame_rate_max)); printf("FrameRate Max: \t\t%d\n", frame_rate_max);

    // check_camera_errors(EVT_CameraSetUInt32Param(camera, "FrameRate",
    // camera_params->frame_rate)); printf("FrameRate Set to: \t%d\n",
    // camera_params.frame_rate);
    update_frame_rate_value(camera, camera_params->frame_rate, camera_params);
    update_focus_value(camera, camera_params->focus, camera_params);
    update_iris_value(camera, camera_params->iris, camera_params);
}

void update_camera_params(Emergent::CEmergentCamera *camera,
                          GigEVisionDeviceInfo *device_info,
                          CameraParams *camera_params) {
    camera_params->gpu_direct = false;
    camera_params->gpu_id = 0;
    check_camera_errors(EVT_CameraOpen(camera, device_info),
                        camera_params->camera_serial.c_str());
    configure_factory_defaults(camera, camera_params);
    unsigned int width_max, height_max;
    check_camera_errors(
        Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max),
        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "Height", &camera_params->height_max);
    EVT_CameraGetUInt32ParamMin(camera, "Height", &camera_params->height_min);
    EVT_CameraGetUInt32ParamInc(camera, "Height", &camera_params->height_inc);
    check_camera_errors(
        Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max),
        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "Width", &camera_params->width_max);
    EVT_CameraGetUInt32ParamMin(camera, "Width", &camera_params->width_min);
    EVT_CameraGetUInt32ParamInc(camera, "Width", &camera_params->width_inc);
    printf("Resolution: \t\t%d x %d\n", width_max, height_max);
    camera_params->width = width_max;
    camera_params->height = height_max;
    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "FrameRate", &camera_params->frame_rate),
                        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "FrameRate",
                                &camera_params->frame_rate_max);
    EVT_CameraGetUInt32ParamMin(camera, "FrameRate",
                                &camera_params->frame_rate_min);
    EVT_CameraGetUInt32ParamInc(camera, "FrameRate",
                                &camera_params->frame_rate_inc);

    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "Exposure", &camera_params->exposure),
                        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "Exposure",
                                &camera_params->exposure_max);
    EVT_CameraGetUInt32ParamMin(camera, "Exposure",
                                &camera_params->exposure_min);
    EVT_CameraGetUInt32ParamInc(camera, "Exposure",
                                &camera_params->exposure_inc);
    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "Gain", &camera_params->gain),
                        camera_params->camera_serial.c_str());
    std::cout << "Gain: " << camera_params->gain << std::endl;
    EVT_CameraGetUInt32ParamMax(camera, "Gain", &camera_params->gain_max);
    std::cout << "Gain max: " << camera_params->gain_max << std::endl;
    EVT_CameraGetUInt32ParamMin(camera, "Gain", &camera_params->gain_min);
    EVT_CameraGetUInt32ParamInc(camera, "Gain", &camera_params->gain_inc);
    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "Iris", &camera_params->iris),
                        camera_params->camera_serial.c_str());
    std::cout << "Iris: " << camera_params->iris << std::endl;
    EVT_CameraGetUInt32ParamMax(camera, "Iris", &camera_params->iris_max);
    std::cout << "Iris max: " << camera_params->iris_max << std::endl;
    EVT_CameraGetUInt32ParamMin(camera, "Iris", &camera_params->iris_min);
    std::cout << "Iris min: " << camera_params->iris_min << std::endl;
    EVT_CameraGetUInt32ParamInc(camera, "Iris", &camera_params->iris_inc);
    std::cout << "Iris inc: " << camera_params->iris_inc << std::endl;
    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "Focus", &camera_params->focus),
                        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "Focus", &camera_params->focus_max);
    EVT_CameraGetUInt32ParamMin(camera, "Focus", &camera_params->focus_min);
    EVT_CameraGetUInt32ParamInc(camera, "Focus", &camera_params->focus_inc);

    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "OffsetY", &camera_params->offsety),
                        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "OffsetY", &camera_params->offsety_max);
    EVT_CameraGetUInt32ParamMin(camera, "OffsetY", &camera_params->offsety_min);
    EVT_CameraGetUInt32ParamInc(camera, "OffsetY", &camera_params->offsety_inc);

    check_camera_errors(Emergent::EVT_CameraGetUInt32Param(
                            camera, "OffsetX", &camera_params->offsetx),
                        camera_params->camera_serial.c_str());
    EVT_CameraGetUInt32ParamMax(camera, "OffsetX", &camera_params->offsetx_max);
    EVT_CameraGetUInt32ParamMin(camera, "OffsetX", &camera_params->offsetx_min);
    EVT_CameraGetUInt32ParamInc(camera, "OffsetX", &camera_params->offsetx_inc);

    const unsigned long enum_buffer_size = 1000;
    unsigned long enum_buffer_size_return = 0;
    char enumBuffer[enum_buffer_size];

    EVT_CameraGetEnumParamRange(camera, "PixelFormat", enumBuffer,
                                enum_buffer_size, &enum_buffer_size_return);
    std::cout << "PixelFormat: " << enumBuffer << std::endl;
    char *enum_member = strtok_s(enumBuffer, ",", &next_token);
    check_camera_errors(
        EVT_CameraSetEnumParam(camera, "PixelFormat", enum_member),
        camera_params->camera_serial.c_str());
    camera_params->pixel_format = std::string(enum_member);

    if (camera_params->pixel_format == "Mono8") {
        camera_params->color = false;
    } else {
        camera_params->color = true;
    }

    if (camera_params->color) {
        EVT_CameraGetEnumParamRange(camera, "ColorTemp", enumBuffer,
                                    enum_buffer_size, &enum_buffer_size_return);
        std::cout << "ColorTemp: " << enumBuffer << std::endl;
        char *enum_member = strtok_s(enumBuffer, ",", &next_token);
        check_camera_errors(
            EVT_CameraSetEnumParam(camera, "ColorTemp", enum_member),
            camera_params->camera_serial.c_str());
        camera_params->color_temp = std::string(enum_member);
    }
}

void camera_trigger_mode(Emergent::CEmergentCamera *camera,
                         CameraParams *camera_params) {
    check_camera_errors(
        EVT_CameraSetEnumParam(camera, "AcquisitionMode", "MultiFrame"),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 76),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        EVT_CameraSetEnumParam(camera, "TriggerSelector", "FrameStart"),
        camera_params->camera_serial.c_str());
    check_camera_errors(EVT_CameraSetEnumParam(camera, "TriggerMode", "On"),
                        camera_params->camera_serial.c_str());
}

// **********************************************sync*****************************************************
void ptp_camera_sync(Emergent::CEmergentCamera *camera,
                     CameraParams *camera_params) {
    // ptp triggering configuration settings
    check_camera_errors(
        EVT_CameraSetEnumParam(camera, "TriggerSource", "Software"),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        EVT_CameraSetEnumParam(camera, "AcquisitionMode", "MultiFrame"),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1),
        camera_params->camera_serial.c_str());
    check_camera_errors(EVT_CameraSetEnumParam(camera, "TriggerMode", "On"),
                        camera_params->camera_serial.c_str());
    check_camera_errors(EVT_CameraSetEnumParam(camera, "PtpMode", "TwoStep"),
                        camera_params->camera_serial.c_str());
}

void ptp_sync_off(Emergent::CEmergentCamera *camera,
                  CameraParams *camera_params) {
    check_camera_errors(Emergent::EVT_CameraSetEnumParam(
                            camera, "AcquisitionMode", "Continuous"),
                        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1),
        camera_params->camera_serial.c_str());
    check_camera_errors(Emergent::EVT_CameraSetEnumParam(
                            camera, "TriggerSelector", "AcquisitionStart"),
                        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetEnumParam(camera, "TriggerMode", "Off"),
        camera_params->camera_serial.c_str());
    check_camera_errors(
        Emergent::EVT_CameraSetEnumParam(camera, "TriggerSource", "Software"),
        camera_params->camera_serial.c_str());
}

// use one camera to get the PTP time, TODO: use linux to get current GMT time
// in seconds
unsigned long long get_current_PTP_time(Emergent::CEmergentCamera *camera) {

    char ptp_status[100];
    unsigned long ptp_status_sz_ret;
    unsigned int ptp_time_high, ptp_time_low;
    // need to open the camera to get ptp time?
    EVT_CameraGetEnumParam(camera, "PtpStatus", ptp_status, sizeof(ptp_status),
                           &ptp_status_sz_ret);
    printf("PTP Status: %s\n", ptp_status);

    // Get and print current time.
    EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch");
    EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high);
    EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low);
    unsigned long long ptp_time =
        (((unsigned long long)(ptp_time_high)) << 32) |
        ((unsigned long long)(ptp_time_low));
    return ptp_time;
}

// test GPO by toggling polarity in manual mode, after open camera, before open
// streaming
void test_gpo_manual_toggle(Emergent::CEmergentCamera *camera) {
    unsigned int count;
    char gpo_str[20];
    bool gpo_polarity = 1;

    // Test GPOs by toggling polarity in manual mode.
    for (count = 0; count < 4; count++) {

        sprintf(gpo_str, "GPO_%d_Mode", count);
        EVT_CameraSetEnumParam(camera, gpo_str, "GPO");
    }

    for (count = 0; count < 4; count++) {
        printf("Toggling GPO %d\t\t", count);
        sprintf(gpo_str, "GPO_%d_Polarity", count);

        for (int blink_count = 0; blink_count < 20; blink_count++) {
            EVT_CameraSetBoolParam(camera, gpo_str, gpo_polarity);
            gpo_polarity = !gpo_polarity;
            usleep(100 * 1000);
            printf(".");
            fflush(stdout);
        }
        printf("\n");
    }
}

void close_camera(Emergent::CEmergentCamera *camera,
                  CameraParams *camera_params) {
    check_camera_errors(EVT_CameraClose(camera),
                        camera_params->camera_serial.c_str());
    printf("\nClose Camera: \t\tCamera Closed\n");
}

void set_frame_buffer(Emergent::CEmergentFrame *evt_frame,
                      CameraParams *camera_params) {
    // Three params used for memory allocation. Worst case covers all models so
    // no recompilation required.
    evt_frame->size_x = camera_params->width;
    evt_frame->size_y = camera_params->height;

    std::string pixel_format = camera_params->pixel_format;
    if (pixel_format == "BayerRG8") {
        evt_frame->pixel_type = GVSP_PIX_BAYRG8;
    } else if (pixel_format == "RGB8Packed") {
        evt_frame->pixel_type = GVSP_PIX_RGB8;
    } else if (pixel_format == "BGR8Packed") {
        evt_frame->pixel_type = GVSP_PIX_BGR8;
    } else if (pixel_format == "YUV411Packed") {
        evt_frame->pixel_type = GVSP_PIX_YUV411_PACKED;
    } else if (pixel_format == "YUV422Packed") {
        evt_frame->pixel_type = GVSP_PIX_YUV422_PACKED;
    } else if (pixel_format == "YUV444Packed") {
        evt_frame->pixel_type = GVSP_PIX_YUV444_PACKED;
    } else if (pixel_format == "BayerGB8") {
        evt_frame->pixel_type = GVSP_PIX_BAYGB8;
    } else // Good for default case which covers color and mono as same size
           // bytes/pixel.
    {      // Note that these settings are used for memory alloc only.
        evt_frame->pixel_type = GVSP_PIX_MONO8;
    }
}

void camera_open_stream(Emergent::CEmergentCamera *camera,
                        CameraParams *camera_params) {
    check_camera_errors(EVT_CameraOpenStream(camera),
                        camera_params->camera_serial.c_str());
}

void allocate_frame_buffer(Emergent::CEmergentCamera *camera,
                           Emergent::CEmergentFrame *evt_frame,
                           CameraParams *camera_params, int buffer_size) {
    for (int frame_count = 0; frame_count < buffer_size; frame_count++) {
        set_frame_buffer(&evt_frame[frame_count], camera_params);
        check_camera_errors(EVT_AllocateFrameBuffer(camera,
                                                    &evt_frame[frame_count],
                                                    EVT_FRAME_BUFFER_ZERO_COPY),
                            camera_params->camera_serial.c_str());
        check_camera_errors(
            EVT_CameraQueueFrame(camera, &evt_frame[frame_count]),
            camera_params->camera_serial.c_str());
    }
}

void allocate_frame_reorder_buffer(Emergent::CEmergentCamera *camera,
                                   Emergent::CEmergentFrame *frame_reorder,
                                   CameraParams *camera_params) {
    set_frame_buffer(frame_reorder, camera_params);
    frame_reorder->convertColor = EVT_COLOR_CONVERT_NONE;
    frame_reorder->convertBitDepth = EVT_CONVERT_NONE;
    check_camera_errors(EVT_AllocateFrameBuffer(camera, frame_reorder,
                                                EVT_FRAME_BUFFER_DEFAULT),
                        camera_params->camera_serial.c_str());
}

void destroy_frame_buffer(Emergent::CEmergentCamera *camera,
                          Emergent::CEmergentFrame *evt_frame, int buffer_size,
                          CameraParams *camera_params) {
    for (int frame_count = 0; frame_count < buffer_size; frame_count++) {
        check_camera_errors(
            EVT_ReleaseFrameBuffer(camera, &evt_frame[frame_count]),
            camera_params->camera_serial.c_str());
    }

    // Host side tear down for stream.
    // EVT_CameraCloseStream(camera);
}

// Use this function with caution, need to reintiate the GigEVisionDeviceInfo
// after changing the camera ip. non persistent
void change_camera_ip(GigEVisionDeviceInfo *device_info, const char *new_ip,
                      CameraParams *camera_params) {
    const char *mac_address = device_info->macAddress;
    const char *subnet_mask = device_info->currentSubnetMask;
    const char *default_gateway = device_info->defaultGateway;
    check_camera_errors(Emergent::EVT_ForceIPEx(mac_address, new_ip,
                                                subnet_mask, default_gateway),
                        camera_params->camera_serial.c_str());
}

// Use this function with caution, need to reintiate the GigEVisionDeviceInfo
// after changing the camera ip.
void change_camera_ip_persistent(GigEVisionDeviceInfo *device_info,
                                 Emergent::CEmergentCamera *camera,
                                 const char *new_ip,
                                 CameraParams *camera_params) {
    const char *mac_address = device_info->macAddress;
    const char *subnet_mask = device_info->currentSubnetMask;
    const char *default_gateway = device_info->defaultGateway;
    check_camera_errors(Emergent::EVT_IPConfig(camera, true, new_ip,
                                               subnet_mask, default_gateway),
                        camera_params->camera_serial.c_str());
}

void quick_print_camera(GigEVisionDeviceInfo *device_info, int camera_idx) {
    std::cout << "camera: " << camera_idx
              << ", serialNumber: " << device_info[camera_idx].serialNumber
              << ", currentIp: " << device_info[camera_idx].currentIp
              << ", nicIp: " << device_info[camera_idx].nic.ip4Address
              << std::endl;
}

int scan_cameras(int max_cameras, GigEVisionDeviceInfo *device_info) {
    unsigned int listcam_buf_size = max_cameras;
    unsigned int count;

    Emergent::EVT_ListDevices(device_info, &listcam_buf_size, &count);
    if (count == 0) {
        printf("Enumerate Cameras: \tNo cameras found.\n");
        return 0;
    } else {
        return count;
    }
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

void sort_cameras_ip(GigEVisionDeviceInfo *device_info,
                     GigEVisionDeviceInfo *sorted_device_info, int cam_count) {
    std::vector<std::string> camera_ips;
    for (int i = 0; i < cam_count; i++) {
        camera_ips.push_back(std::string(device_info[i].currentIp));
    }

    int j = 0;
    for (auto i : sort_indexes(camera_ips)) {
        sorted_device_info[j] = device_info[i];
        j++;
    }
}

int order_for_test_rig(int max_cameras, GigEVisionDeviceInfo *device_info,
                       GigEVisionDeviceInfo *ordered_device_info) {
    int cameras_found = 0;
    unsigned int listcam_buf_size = max_cameras;
    unsigned int count;

    Emergent::EVT_ListDevices(device_info, &listcam_buf_size, &count);

    if (count == 0) {
        printf("Enumerate Cameras: \tNo cameras found. Exiting program.\n");
        return 0;
    } else {
        for (unsigned int i = 0; i < count; i++) {

            if (strcmp(device_info[i].serialNumber, "2002490") == 0) {
                ordered_device_info[0] = device_info[i];
            } else if (strcmp(device_info[i].serialNumber, "2002496") == 0) {
                ordered_device_info[1] = device_info[i];
            } else if (strcmp(device_info[i].serialNumber, "2002488") == 0) {
                ordered_device_info[2] = device_info[i];
            } else if (strcmp(device_info[i].serialNumber, "2002489") == 0) {
                ordered_device_info[3] = device_info[i];
            }

            // center one
            else if (strcmp(device_info[i].serialNumber, "2002494") == 0) {
                ordered_device_info[4] = device_info[i];
            }
        }

        printf("Found %d cameras. \n", count);
        for (unsigned int i = 0; i < count; i++) {
            quick_print_camera(device_info, i);
        }
        return count;
    }
}
