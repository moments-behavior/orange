#include "camera_driver_helper.h"

string get_evt_error_string(EVT_ERROR error)
{
    string error_string; 
    switch (error)
    {
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
            error_string = "Camera lost connection due to disconnected, powered off, crashing etc.";
            break;
        case EVT_ERROR_GENICAM_ERROR:
            error_string = "Generic GeniCam error from GeniCam lib.";
            break;
        case EVT_ERROR_GENICAM_NOT_MATCH:
            error_string =  "Parameter not matched.";
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
            error_string = "SDK error, should not occur. Can be removed if sdk is proved to be correct.";
            break;
        case EVT_GENERAL_ERROR:
            error_string = "General error.";
            break;
    }
    return error_string;
}

