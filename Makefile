# Linux:
#   apt-get install libglfw-dev

CXX = g++
DIR_OUT = targets
$(shell   mkdir -p $(DIR_OUT))


DIR_SRC = ./src
EXE = $(DIR_OUT)/orange


EMERGENT_DIR = /opt/EVT/eSDK
CUDA_DIR = /usr/local/cuda-11.4
NVENC = ./third_party/NvEncoder
IMGUI_DIR = ./third_party/imgui
OTHER_LIB = ./third_party/other_lib


DIR_INC = -I$(DIR_SRC)
DIR_INC += -I$(OTHER_LIB)
DIR_INC += -I$(EMERGENT_DIR)/include
DIR_INC += -I$(NVENC)/include -I$(NVENC) 
DIR_INC += -I/usr/include/opencv4 -I/usr/lib/x86_64-linux-gnu/gstreamer-1.0 -I$(CUDA_DIR)/include
DIR_INC += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends

SOURCES = $(wildcard $(DIR_SRC)/*.cpp) 
SOURCES += $(wildcard $(NVENC)/*.cpp) 
SOURCES += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SOURCES += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

SOURCES_NO_DIR = $(notdir $(SOURCES))
OBJS_CXX = $(patsubst %.cpp,$(DIR_OUT)/%.o,$(SOURCES_NO_DIR)) 


CXXFLAGS += -g -Ofast -ffast-math  -std=c++11

LDLIBS_FLAGS += -lGLEW -lGLU
LDLIBS_FLAGS += -lGL `pkg-config --static --libs glfw3`
LDLIBS_FLAGS += `pkg-config --cflags --libs x11`
LDLIBS_FLAGS += `pkg-config --cflags libavformat libswscale libswresample libavutil libavcodec`
LDLIBS_FLAGS += `pkg-config --libs libavformat libswscale libswresample libavutil libavcodec`
LDLIBS_FLAGS += -L$(CUDA_DIR)/lib64/ -lcudart -lcuda -lnppicc -lnvidia-encode
LDLIBS_FLAGS += -lopencv_core -lopencv_imgcodecs -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio
LDLIBS_FLAGS += -lm -lpthread -lgstreamer-1.0 
LDLIBS_FLAGS += -L$(EMERGENT_DIR)/lib  -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision


$(DIR_OUT)/%.o: $(IMGUI_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(DIR_INC) -c -o $@ $<


$(DIR_OUT)/%.o: $(IMGUI_DIR)/backends/%.cpp
	$(CXX) $(CXXFLAGS) $(DIR_INC) -c -o $@ $<


$(DIR_OUT)/%.o: $(NVENC)/%.cpp
	$(CXX) $(CXXFLAGS) $(DIR_INC) -c -o $@ $<


$(DIR_OUT)/%.o: $(DIR_SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(DIR_INC) -c -o $@ $<


$(EXE): $(OBJS_CXX)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDLIBS_FLAGS)


.PHONY:all
all:  
	$(EXE)


.PHONY:clean
clean:
	sudo rm -rf $(CXXEXE) $(DIR_OUT)

