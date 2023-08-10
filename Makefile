CXX=g++
CFLAGS = -Wall -Wformat -std=c++17

DIR_OUT=./targets
CXXEXE = $(DIR_OUT)/orange

# Cuda config
CUDA_PATH=/usr/local/cuda
CUDA_INC_PATH=$(CUDA_PATH)/include
CUDA_LIB_PATH=$(CUDA_PATH)/lib64
NVCC=$(CUDA_PATH)/bin/nvcc
LIBS_CUDA = -L$(CUDA_LIB_PATH) -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc

# OpenGL config
LIBS_GL=-lGL -lGLEW

#Emergent config
DIR_EMERGENT_INC = $(EMERGENT_DIR)/eSDK/include/ 
LIBS_EMERGENT = -L$(EMERGENT_DIR)/eSDK/lib/  -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision

# IMGUI 
IMGUI_DIR = third_party/imgui

# IMPLOT
IMPLOT_DIR = third_party/implot
IMFILEBROWSER_DIR = third_party/imgui-filebrowser
ICONFONT_DIR= third_party/IconFontCppHeaders

# FFmpeg
FFMPEG_INC = $(HOME)/nvidia/ffmpeg/build/include/
FFMPEG_LIB = -L$(HOME)/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec

DIR_INC = -I$(CUDA_INC_PATH) -I$(DIR_EMERGENT_INC) -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I./src/NvEncoder -I./nvenc_api/include -I$(FFMPEG_INC) -I$(IMPLOT_DIR) -I$(IMFILEBROWSER_DIR) -I$(ICONFONT_DIR)
LDLIBS_FLAGS = -lpthread $(LIBS_EMERGENT) $(LIBS_CUDA) $(LIBS_GL) $(FFMPEG_LIB) 	
LDLIBS_FLAGS += `pkg-config --static --libs glfw3`

SOURCE_CXX = src/orange.cpp src/camera_driver_helper.cpp src/camera.cpp src/video_capture.cpp
SOURCE_CXX += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp 
SOURCE_CXX += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SOURCE_CXX += $(IMPLOT_DIR)/implot.cpp $(IMPLOT_DIR)/implot_items.cpp $(IMPLOT_DIR)/implot_demo.cpp
SOURCE_CXX += $(wildcard src/NvEncoder/*.cpp) 
OBJS_CXX = $(patsubst %.cpp,$(DIR_OUT)/%.o,$(SOURCE_CXX)) 

SOURCE_CU = $(wildcard src/*.cu)
OBJS_CU = $(patsubst %.cu,$(DIR_OUT)/%.cb, $(SOURCE_CU)) 

CFLAGS += `pkg-config --cflags glfw3`

$(DIR_OUT)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) -c $(CFLAGS) $(DIR_INC) $< -o $@

$(DIR_OUT)/%.cb: %.cu
	mkdir -p $(dir $@)
	$(NVCC) --lib -Xcompiler -fPIC $< -o $@


$(CXXEXE) : $(OBJS_CXX) $(OBJS_CU)
	$(CXX) -o $(CXXEXE) $(OBJS_CXX) $(OBJS_CU) $(LDLIBS_FLAGS)

.PHONY:all
all:  $(CXXEXE)

.PHONY:clean
clean:
	rm -fr $(CXXEXE) $(DIR_OUT)


