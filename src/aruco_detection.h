#ifndef ORANGE_ARUCO_DETECTION
#define ORANGE_ARUCO_DETECTION
#include "realtime_tool.h"

void marker_detection_thread(CPURender* cpu_buffers, ArucoMarker2d* marker2d_all_cams, ArucoMarker3d* marker3d, CameraParams* cameras_params, CameraCalibResults* calib_results, int num_cameras)
{
    while (true) {

        if (!marker2d_all_cams->detected_cameras.empty()) {
            marker2d_all_cams->detected_cameras.clear();
            marker2d_all_cams->detected_points.clear();
            marker3d->corners.clear();
        }

        for (int i = 0; i < num_cameras; i++)
        {
            cpu_buffers[i].display_buffer.available_to_write = false;
        }


        for (int i = 0; i < num_cameras; i++) {
            aruco_detection(&cpu_buffers[i].display_buffer, cameras_params, marker2d_all_cams); 
        } 

        if(find_marker3d(marker2d_all_cams, marker3d, calib_results)) {
            // send the frames 
            
            std::cout << "Marker tvec: " << marker3d->t_vec << std::endl;
            std::cout << "Marker normal: " << marker3d->normal << std::endl;
        }

        for (int i = 0; i < num_cameras; i++)
        {
            cpu_buffers[i].display_buffer.available_to_write = true;
        }

    }

}

#endif