cmake_minimum_required(VERSION 3.10)
project(AdEx_SNN_Deployment)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(snn_infer
    src/main.cpp
    src/snn_core.cpp
)