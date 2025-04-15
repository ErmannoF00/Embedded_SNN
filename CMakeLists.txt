# File: CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(gesture_snn)

set(CMAKE_CXX_STANDARD 17)

# Paths
include_directories(cpp_inference)
include_directories(deployment)

add_executable(snn_main 
    cpp_inference/main.cpp 
    cpp_inference/snn_core.cpp
)

add_executable(simulate_input 
    deployment/simulate_input.cpp
)
