cmake_minimum_required(VERSION 3.10)
project(OrangeApp LANGUAGES CXX CUDA)

# CUDA — build for both tested architectures (Turing 7.5, Ampere 8.0) by default.
# Override at configure time with e.g. -DCMAKE_CUDA_ARCHITECTURES=86 for a single arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80)
endif()

# ---- Options ---------------------------------------------------------------
option(BUILD_ORANGE         "Build GUI 'orange' app"        ON)
option(BUILD_CAM_SERVER     "Build cam_server"  ON)
option(BUILD_YOLO_OFFLINE   "Build yolo_offline tool"       ON)

# ---- Dirs ------------------------------------------------------------------
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

# ---- Common includes/links -------------------------------------------------
include_directories(
    ${SRC_DIR}
    ${DIR_NVENCODER}
    ${DIR_NVENC_API}/include
    /opt/EVT/eSDK/include/
    /usr/local/cuda/include/
    ${DIR_FFMPEG}/build/include/
    /usr/local/include/opencv4
    ${THIRD_PARTY_DIR}/flatbuffers/include/
    ${DIR_TENSORRT}/include
)

link_directories(
    /opt/EVT/eSDK/lib/
    /usr/local/cuda/lib64/
    ${DIR_FFMPEG}/build/lib/
    ${DIR_TENSORRT}/lib/
)

# ---- Sources ---------------------------------------------------------------
file(GLOB NVENCODER_SOURCES ${SRC_DIR}/NvEncoder/*.cpp)
set(CUDA_SOURCES ${SRC_DIR}/kernel.cu)

# orange (GUI) sources only if building it
if(BUILD_ORANGE)
    file(GLOB ORANGE_SOURCES
        ${SRC_DIR}/*.cpp
        ${DIR_NVENCODER}/*.cpp
    )
    list(REMOVE_ITEM ORANGE_SOURCES
        ${SRC_DIR}/cam_server.cpp
        ${SRC_DIR}/yolo_offline.cpp
    )

    # ImGui/ImPlot only when GUI target is on
    set(IMGUI_SOURCES
        ${DIR_IMGUI}/imgui.cpp
        ${DIR_IMGUI}/imgui_demo.cpp
        ${DIR_IMGUI}/imgui_draw.cpp
        ${DIR_IMGUI}/imgui_tables.cpp
        ${DIR_IMGUI}/imgui_widgets.cpp
        ${DIR_IMGUI_BACKEND}/imgui_impl_glfw.cpp
        ${DIR_IMGUI_BACKEND}/imgui_impl_opengl3.cpp
    )
    set(IMPLOT_SOURCES
        ${DIR_IMPLOT}/implot.cpp
        ${DIR_IMPLOT}/implot_items.cpp
        ${DIR_IMPLOT}/implot_demo.cpp
    )
    set(FILEDIALOG_SOURCES ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp)
endif()

# cam_server sources
if(BUILD_CAM_SERVER)
    file(GLOB CAM_SERVER_SOURCES
        ${SRC_DIR}/*.cpp
        ${DIR_NVENCODER}/*.cpp
    )
    list(REMOVE_ITEM CAM_SERVER_SOURCES
        ${SRC_DIR}/orange.cpp
        ${SRC_DIR}/yolo_offline.cpp
        ${SRC_DIR}/yolov8_det.cpp
        ${SRC_DIR}/opengldisplay.cpp
        ${SRC_DIR}/FrameDetector.cpp
        ${SRC_DIR}/realtime_tool.cpp
        ${SRC_DIR}/detect3d.cpp
        ${SRC_DIR}/gui.cpp
        ${SRC_DIR}/gx_helper.cpp
        ${SRC_DIR}/host_client_imgui.cpp
    )
endif()

# yolo_offline sources
if(BUILD_YOLO_OFFLINE)
    set(YOLO_OFFLINE_SOURCES
        ${SRC_DIR}/yolo_offline.cpp
        ${SRC_DIR}/yolov8_det.cpp
        ${CUDA_SOURCES}
    )
endif()

# ---- Targets ---------------------------------------------------------------
if(BUILD_ORANGE)
    add_executable(orange
        ${ORANGE_SOURCES}
        ${IMGUI_SOURCES}
        ${IMPLOT_SOURCES}
        ${FILEDIALOG_SOURCES}
        ${CUDA_SOURCES}
    )
    target_compile_definitions(orange PRIVATE)
    # GUI-only deps
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED glfw3)
    target_include_directories(orange PRIVATE
        ${DIR_IMGUI} ${DIR_IMGUI_BACKEND} ${DIR_IMPLOT} ${DIR_FILEBROWSER} ${DIR_ICONFONT}
        ${GLFW3_INCLUDE_DIRS}
    )
    target_link_libraries(orange
        pthread m enet GLEW GL
        cudart cuda nppicc nppidei nvidia-encode nppc nppig nppial
        EmergentCamera EmergentGenICam EmergentGigEVision
        avformat swscale swresample avutil avcodec
        opencv_sfm opencv_core opencv_imgcodecs opencv_imgproc opencv_calib3d
        nvinfer nvinfer_plugin nvToolsExt glfw
        ${GLFW3_LIBRARIES}
    )
endif()

if(BUILD_CAM_SERVER)
    add_executable(cam_server
        ${CAM_SERVER_SOURCES}
        ${CUDA_SOURCES}
    )
    target_compile_definitions(cam_server PRIVATE HEADLESS)
    target_link_libraries(cam_server
        EmergentCamera EmergentGenICam EmergentGigEVision
        pthread enet
        cudart cuda nppicc nppidei nvidia-encode nppc
        avformat swscale swresample avutil avcodec m
        opencv_sfm opencv_core opencv_imgcodecs opencv_imgproc
    )
endif()

if(BUILD_YOLO_OFFLINE)
    add_executable(yolo_offline ${YOLO_OFFLINE_SOURCES})
    target_link_libraries(yolo_offline
        opencv_sfm opencv_core opencv_imgproc opencv_videoio opencv_highgui opencv_video
        nvinfer nvinfer_plugin
        cudart cuda nppicc nppidei nvidia-encode nppc nppig nppial
    )
endif()
