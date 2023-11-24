cmake_minimum_required(VERSION 3.26)
project(gpt_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/opt/homebrew/lib/python3.11/site-packages/torch/share/cmake")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cpp
        include/gpt.h
        src/gpt.cpp)

target_include_directories(gpt_cpp PRIVATE include)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)