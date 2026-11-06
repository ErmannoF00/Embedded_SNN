cmake_minimum_required(VERSION 3.16)
project(embedded_snn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(SNN_SANITIZERS "Host address/undefined-behavior sanitizers" OFF)
if(SNN_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(snn_runtime STATIC src/snn_runtime.cpp)
target_include_directories(snn_runtime PUBLIC include generated)
target_compile_options(snn_runtime PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
# The inference library does not need heap allocation, exception machinery or RTTI.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(snn_runtime PRIVATE -fno-exceptions -fno-rtti)
endif()

add_executable(snn_host src/host_cli.cpp)
target_link_libraries(snn_host PRIVATE snn_runtime)
target_compile_options(snn_host PRIVATE -Wall -Wextra -Wpedantic)

add_executable(snn_selftest src/snn_selftest.cpp)
target_link_libraries(snn_selftest PRIVATE snn_runtime)

add_executable(snn_benchmark src/snn_benchmark.cpp)
target_link_libraries(snn_benchmark PRIVATE snn_runtime)

enable_testing()
add_test(NAME snn_selftest COMMAND snn_selftest)
