add_library (third_party INTERFACE)
target_include_directories (third_party INTERFACE ${CMAKE_CURRENT_LIST_DIR})

add_library (imgui INTERFACE)
set(IMGUI_DIR ${CMAKE_CURRENT_LIST_DIR}/imgui)
target_include_directories (imgui INTERFACE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_sources (imgui INTERFACE
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)

add_library (IconFontCppHeaders INTERFACE)
target_include_directories (IconFontCppHeaders INTERFACE ${CMAKE_CURRENT_LIST_DIR}/IconFontCppHeaders)

add_library (eSDK INTERFACE)
target_include_directories (eSDK INTERFACE $ENV{EMERGENT_DIR}/eSDK/include)
target_link_directories (eSDK INTERFACE $ENV{EMERGENT_DIR}/eSDK/lib)
target_link_libraries (eSDK INTERFACE EmergentCamera EmergentGigEVision)

add_library (nvcodec INTERFACE)
target_include_directories (nvcodec INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nvcodec)
target_link_directories (nvcodec INTERFACE ${CMAKE_CURRENT_LIST_DIR}/nvcodec/lib)
target_link_libraries (nvcodec INTERFACE nvcuvid nvencodeapi ${CUDA_LIBRARIES})

add_library (ffmpeg INTERFACE)
target_include_directories (ffmpeg INTERFACE ${FFMPEG_INCLUDE_DIRS})
target_link_directories (ffmpeg INTERFACE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries (ffmpeg INTERFACE ${FFMPEG_LIBRARIES})
