cmake_minimum_required(VERSION 3.10)
project(gesture_snn)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

add_executable(snn_main 
    src/main.cpp 
    src/snn_core.cpp
)

add_executable(simulate_input 
    src/simulate_input.cpp
)
