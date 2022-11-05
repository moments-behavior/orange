add_library (camera
    camera.cpp
    video_capture_gpu.cpp
    camera_driver_helper.cpp
    NvEncoder/NvEncoder.cpp
    NvEncoder/NvEncoderCuda.cpp
    NvEncoder/NvEncoderOutputInVidMemCuda.cpp)
target_link_libraries (camera eSDK nvcodec ffmpeg cuda nppicc)

add_executable (orange orange.cpp)
target_link_libraries (orange PRIVATE camera third_party imgui IconFontCppHeaders eSDK nvcodec ffmpeg glfw GLEW::GLEW)