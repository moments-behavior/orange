cmake_minimum_required(VERSION 3.10)
project(OrangeApp LANGUAGES CXX CUDA)

# Set build type
set(CMAKE_BUILD_TYPE Debug)

# Set CUDA architecture and options
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

# Directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
set(DIR_IMGUI ${THIRD_PARTY_DIR}/imgui)
set(DIR_IMGUI_BACKEND ${THIRD_PARTY_DIR}/imgui/backends)
set(DIR_IMPLOT ${THIRD_PARTY_DIR}/implot)
set(DIR_FILEBROWSER ${THIRD_PARTY_DIR}/ImGuiFileDialog)
set(DIR_ICONFONT ${THIRD_PARTY_DIR}/IconFontCppHeaders)
set(DIR_NVENCODER ${SRC_DIR}/NvEncoder)
set(DIR_FFMPEG $ENV{HOME}/nvidia/ffmpeg)
set(DIR_TENSORRT $ENV{HOME}/nvidia/TensorRT)
set(DIR_NVENC_API ${CMAKE_SOURCE_DIR}/nvenc_api)

# Add the source files
file(GLOB ORANGE_SOURCES
    ${SRC_DIR}/*.cpp
    ${DIR_NVENCODER}/*.cpp
)

list(REMOVE_ITEM ORANGE_SOURCES ${SRC_DIR}/orange_headless_client.cpp)

file(GLOB NVENCODER_SOURCES ${PROJECT_SOURCE_DIR}/src/NvEncoder/*.cpp)
# Add sourc files for orange_client
file(GLOB ORANGE_CLIENT_SOURCES
    ${SRC_DIR}/*.cpp
    ${DIR_NVENCODER}/*.cpp
)
list(REMOVE_ITEM ORANGE_CLIENT_SOURCES ${SRC_DIR}/orange.cpp ${SRC_DIR}/yolov8_det.cpp ${SRC_DIR}/opengldisplay.cpp ${SRC_DIR}/FrameDetector.cpp ${SRC_DIR}/realtime_tool.cpp ${SRC_DIR}/detect3d.cpp)

set(YOLO_OFFLINE_SOURCES
    benchmark/yolo_offline.cpp
    src/yolov8_det.cpp
)

# ImGui Sources
set(IMGUI_SOURCES
    ${DIR_IMGUI}/imgui.cpp
    ${DIR_IMGUI}/imgui_demo.cpp
    ${DIR_IMGUI}/imgui_draw.cpp
    ${DIR_IMGUI}/imgui_tables.cpp
    ${DIR_IMGUI}/imgui_widgets.cpp
    ${DIR_IMGUI_BACKEND}/imgui_impl_glfw.cpp
    ${DIR_IMGUI_BACKEND}/imgui_impl_opengl3.cpp
)

# ImPlot Sources
set(IMPLOT_SOURCES
    ${DIR_IMPLOT}/implot.cpp
    ${DIR_IMPLOT}/implot_items.cpp
    ${DIR_IMPLOT}/implot_demo.cpp
)

# FileDialog Source
set(FILEDIALOG_SOURCES
    ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
)

# Add CUDA sources
set(CUDA_SOURCES
    ${SRC_DIR}/kernel.cu
)

# Add include directories
include_directories(
    ${SRC_DIR}
    ${DIR_IMGUI}
    ${DIR_IMGUI_BACKEND}
    ${DIR_IMPLOT}
    ${DIR_FILEBROWSER}
    ${DIR_ICONFONT}
    ${DIR_NVENCODER}
    ${DIR_NVENC_API}/include
    /opt/EVT/eSDK/include/
    /usr/local/cuda/include/
    ${DIR_FFMPEG}/build/include/
    /usr/local/include/opencv4
    ${THIRD_PARTY_DIR}/flatbuffers/include/
    ${DIR_TENSORRT}/include
)

# Add link directories
link_directories(
    /opt/EVT/eSDK/lib/
    /usr/local/cuda/lib64/
    ${DIR_FFMPEG}/build/lib/
    ${DIR_TENSORRT}/lib/
)

# Define target
add_executable(orange
    ${ORANGE_SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
    ${FILEDIALOG_SOURCES}
    ${CUDA_SOURCES}
)

add_executable(orange_client
    ${ORANGE_CLIENT_SOURCES}
    ${CUDA_SOURCES}
)

add_executable(yolo_offline
    ${YOLO_OFFLINE_SOURCES}
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(orange
    pthread
    m
    enet
    GLEW
    GL
    cudart
    cuda
    nppicc
    nppidei
    nvidia-encode
    nppc
    nppig
    nppial
    EmergentCamera
    EmergentGenICam
    EmergentGigEVision
    avformat
    swscale
    swresample
    avutil
    avcodec
    opencv_sfm
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_calib3d
    nvinfer
    nvinfer_plugin
    glfw
)

# Link libraries
target_link_libraries(orange_client
    EmergentCamera
    EmergentGenICam
    EmergentGigEVision
    pthread
    enet
    cudart
    cuda
    nppicc
    nppidei
    nvidia-encode
    nppc
    avformat
    swscale
    swresample
    avutil
    avcodec
    m
    opencv_sfm
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)
target_compile_definitions(orange_client PRIVATE HEADLESS)

target_link_libraries(yolo_offline
    opencv_sfm
    opencv_core
    opencv_imgproc
    opencv_videoio
    opencv_highgui
    opencv_video
    nvinfer
    nvinfer_plugin
    cudart
    cuda
    nppicc
    nppidei
    nvidia-encode
    nppc
    nppig
    nppial
)

# Link pkg-config libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)
target_include_directories(orange PRIVATE ${GLFW3_INCLUDE_DIRS})
target_link_libraries(orange ${GLFW3_LIBRARIES})

# Add output directory
#set_target_properties(orange PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/targets)
#set_target_properties(orange PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/targets)
