cmake_minimum_required(VERSION 3.10)
project(OrangeApp LANGUAGES CXX CUDA)

# CUDA — build for both tested architectures (Turing 7.5, Ampere 8.0) by default.
# Override at configure time with e.g. -DCMAKE_CUDA_ARCHITECTURES=86 for a single arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80)
endif()

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
set(DIR_NVENC_API ${CMAKE_SOURCE_DIR}/nvenc_api)

# ---- Common includes/links -------------------------------------------------
include_directories(
    ${SRC_DIR}
    ${DIR_NVENCODER}
    ${DIR_NVENC_API}/include
    /opt/EVT/eSDK/include/
    /usr/local/cuda/include/
    ${DIR_FFMPEG}/build/include/
)

link_directories(
    /opt/EVT/eSDK/lib/
    /usr/local/cuda/lib64/
    ${DIR_FFMPEG}/build/lib/
)

# ---- Sources ---------------------------------------------------------------
set(CUDA_SOURCES ${SRC_DIR}/kernel.cu)

# All remaining src/*.cpp belong to the single GUI recorder target.
file(GLOB ORANGE_SOURCES
    ${SRC_DIR}/*.cpp
    ${DIR_NVENCODER}/*.cpp
)

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

# ---- Target: orange (GUI recorder) -----------------------------------------
add_executable(orange
    ${ORANGE_SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
    ${FILEDIALOG_SOURCES}
    ${CUDA_SOURCES}
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)
target_include_directories(orange PRIVATE
    ${DIR_IMGUI} ${DIR_IMGUI_BACKEND} ${DIR_IMPLOT} ${DIR_FILEBROWSER} ${DIR_ICONFONT}
    ${GLFW3_INCLUDE_DIRS}
)
target_link_libraries(orange
    pthread m GLEW GL glfw
    cudart cuda nppicc nppidei nppc nppig nppial nvidia-encode
    EmergentCamera EmergentGenICam EmergentGigEVision
    avformat swscale swresample avutil avcodec
    # NVTX3 (nvtx3/nvToolsExt.h) is header-only on CUDA 12.x and 13 — no nvToolsExt link.
    ${GLFW3_LIBRARIES}
)

# Emit DT_RPATH instead of DT_RUNPATH: transitively-needed bundled FFmpeg libs
# (e.g. libswresample.so.3, pulled in by libavcodec.so.58) live in ~/nvidia/ffmpeg/build/lib.
# RUNPATH only covers orange's *direct* deps, so the loader can't find transitive ones;
# RPATH covers the whole dependency chain.
set_target_properties(orange PROPERTIES LINK_FLAGS "-Wl,--disable-new-dtags")
