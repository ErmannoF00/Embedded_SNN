cmake_minimum_required(VERSION 3.13)
project(AdEx_SNN_Embedded C CXX ASM)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-as)

set(CMAKE_C_FLAGS "-mcpu=cortex-m4 -mthumb -O2")
set(CMAKE_CXX_FLAGS "-mcpu=cortex-m4 -mthumb -O2 -fno-exceptions -fno-rtti")
set(CMAKE_EXE_LINKER_FLAGS "-T${CMAKE_SOURCE_DIR}/linker.ld -Wl,--gc-sections --specs=rdimon.specs -lc -lrdimon")

include_directories(${PROJECT_SOURCE_DIR}/src)

set_source_files_properties(src/startup_stm32f4xx.s PROPERTIES LANGUAGE ASM)

add_executable(snn_embedded
    src/embedded_main.cpp
    src/snn_core.cpp
)

set_target_properties(snn_embedded PROPERTIES OUTPUT_NAME "snn_embedded.elf")

add_custom_command(TARGET snn_embedded POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary snn_embedded.elf snn_embedded.bin
    COMMENT "Generating binary file"
)
