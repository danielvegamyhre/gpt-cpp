cmake_minimum_required(VERSION 3.26)
project(gpt_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# pytorch
set(CMAKE_PREFIX_PATH "/opt/homebrew/lib/python3.11/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

# sentencepiece
find_library(SP_LIBRARY NAMES sentencepiece)
find_library(SP_TRAIN_LIBRARY NAMES sentencepiece_train)
find_path(SP_INCLUDE_DIR NAMES sentencepiece_processor.h)
find_path(SP_INCLUDE_DIR NAMES sentencepiece_trainer.h)

add_executable(${PROJECT_NAME} src/main.cpp
        include/gpt.h
        src/gpt.cpp
        src/train_tokenizer.cpp)

target_include_directories(gpt_cpp PRIVATE include)
target_include_directories(gpt_cpp PRIVATE ${SP_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${SP_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${SP_TRAIN_LIBRARY})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)