# Linux:
#   apt-get install libglfw-dev

CXX = g++
DIR_OUT = ./targets
EXE = $(DIR_OUT)/orange


EMERGENT_DIR = /opt/EVT/eSDK
CUDA_DIR = /usr/local/cuda-11.4
NVENC = ./third_party/NvEncoder
IMGUI_DIR = ./third_party/imgui


DIR_INC = -I$(EMERGENT_DIR)/include
DIR_INC += -I$(NVENC)/include -I$(NVENC) 
DIR_INC += -I/usr/include/opencv4 -I/usr/lib/x86_64-linux-gnu/gstreamer-1.0 -I$(CUDA_DIR)/include
DIR_INC += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends

SOURCES = $(wildcard ./src/*.cpp) 
SOURCES += $(wildcard ./third_party/NvEncoder/*.cpp)
SOURCES += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SOURCES += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
OBJS_CXX = $(patsubst %.cpp,$(DIR_OUT)/%.o,$(SOURCES))) 



CXXFLAGS += -g -Wall -Wformat -std=c++11


LDLIBS_FLAGS += -lGL `pkg-config --static --libs glfw3`
LDLIBS_FLAGS += `pkg-config --cflags --libs x11`
LDLIBS_FLAGS += `pkg-config --cflags libavformat libswscale libswresample libavutil libavcodec`
LDLIBS_FLAGS += `pkg-config --libs libavformat libswscale libswresample libavutil libavcodec`
LDLIBS_FLAGS += -lGLEW -lGLU
LDLIBS_FLAGS += -L$(CUDA_DIR)/lib64/ -lcudart -lcuda -lnppicc -lnvidia-encode
LDLIBS_FLAGS += -lopencv_core -lopencv_imgcodecs -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio
LDLIBS_FLAGS += -lm -lpthread -lgstreamer-1.0 
LDLIBS_FLAGS += -L$(EMERGENT_DIR)/lib  -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision


$(DIR_OUT)/%.o: %.cpp
	mkdir -p $(DIR_OUT)
	$(CXX) -c $(CXXFLAGS) $(DIR_INC) $< -o $@

$(EXE): $(OBJS_CXX)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDLIBS_FLAGS)


.PHONY:all
all:  $(EXE)



