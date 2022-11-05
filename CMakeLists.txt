# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 3.21)
project (orange)

enable_language (CUDA)
find_package (CUDA 11.6 REQUIRED)
find_package (glfw3 CONFIG REQUIRED)
find_package (GLEW REQUIRED)
find_package (FFMPEG REQUIRED)

# Third party dependencies are described here
add_subdirectory (third_party)

# The compiled library code is here
add_subdirectory (src)