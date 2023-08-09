cmake_minimum_required(VERSION 3.17)
project(onnx2trt CXX CUDA)  # 使CMake可以识别并验证CUDA编译器nvcc

enable_language(CUDA CXX) #激活CUDA语言支持，使用第一个写法时要进行注释  需要在CMAKE_CUDA_COMPILER中指定nvcc路径
# 要检查 CUDA 是否可用，可使用 CheckLanuage:
include(CheckLanguage)
check_language(CUDA)

message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")
message("CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
# 若cuda可选 可以enable_language(CUDA)

# 设置语言标准
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# CMake 中许多名称中带有 CXX 的变量都有 CUDA 版本。例如，要设置 CUDA 所需的 C++ 标准
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")

# 设置libtorch地址
# set(CMAKE_PREFIX_PATH "/home/rui.bai/download/libtorch")
set(CMAKE_PREFIX_PATH "/home/br/program/libtorch"
                      ${CMAKE_PREFIX_PATH})
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if (NOT Torch_FOUND)
    message(FATAL_ERROR "Libtorch Not Found!")
endif (NOT Torch_FOUND)
message(STATUS "Libtorch status: ")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "Torch_FOUND: ${Torch_FOUND}")
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

set(CMAKE_FIND_DEBUG_MODE FALSE)
# set(CUDAToolkit_ROOT "/home/rui.bai/cuda-11.3")
set(CUDAToolkit_ROOT "/usr/local/cuda-11.3")
find_package(CUDAToolkit REQUIRED)
# find_library(CUDART_LIBRARY CUDA::cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
set(CMAKE_FIND_DEBUG_MODE FALSE)
message(STATUS "CUDAToolkit_FOUND ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_BIN_DIR ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDAToolkit_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS}")

# set(TENSORRT_LIB_PATH "/home/rui.bai/download/TensorRT-8.5.3.1/lib")
set(TENSORRT_INCLUDE_DIR "/home/br/program/TensorRT-8.5.3.1/include")
set(TENSORRT_LIB_PATH "/home/br/program/TensorRT-8.5.3.1/lib")
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so*")
file(GLOB CUDAToolkit_LIBS "${CUDAToolkit_LIBRARY_DIR}/*.so*")
message("CUDAToolkit_LIBS: ${CUDAToolkit_LIBS}")

add_executable(${PROJECT_NAME} onnx2trt_engine.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
                           ${CUDAToolkit_INCLUDE_DIRS}
                           ${TENSORRT_INCLUDE_DIR}
                           ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDAToolkit_LIBS}
                                      ${TRT_LIBS}
                                      ${TORCH_LIBRARIES})

# add_executable(test_python_voxel src/test_python_voxel.cpp
#                                  src/voxelization_cuda.cu
#                                  src/voxelization.cpp)
# target_link_libraries(test_python_voxel "${TORCH_LIBRARIES}")


# add_executable(python_voxel src/python_voxel.cpp)
# target_link_libraries(python_voxel "${TORCH_LIBRARIES}" voxel_cuda)
# set_property(TARGET python_voxel PROPERTY CXX_STANDARD 14)

# rm -rf build && mkdir build && cd build/ && cmake -DCMAKE_PREFIX_PATH=/home/mm/bairui/download/libtorch ..
# 设置当前环境下libtorch的位置
# cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
