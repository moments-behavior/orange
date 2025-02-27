#!/bin/bash
DIR="$(dirname "$(realpath "$0")")"
mkdir -p cmake-build-debug
/home/user/Downloads/clion-2024.3.1/bin/cmake/linux/x64/bin/cmake --build $DIR/cmake-build-debug --target all -j $(( $(nproc) - 2 ))