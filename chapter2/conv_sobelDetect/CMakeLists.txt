cmake_minimum_required(VERSION 3.17)
project(sobelDetect CXX CUDA)  # 使CMake可以识别并验证CUDA编译器nvcc

enable_language(CUDA CXX)
include(CheckLanguage)
check_language(CUDA)

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

# set cuda
set(CMAKE_FIND_DEBUG_MODE FALSE)
set(CUDAToolkit_ROOT "/usr/local/cuda-11.3")
find_package(CUDAToolkit REQUIRED)
# find_library(CUDART_LIBRARY CUDA::cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
set(CMAKE_FIND_DEBUG_MODE FALSE)
message(STATUS "CUDAToolkit_FOUND ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_BIN_DIR ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDAToolkit_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS}")

file(GLOB CUDAToolkit_LIBS "${CUDAToolkit_LIBRARY_DIR}/*.so*")
message("CUDAToolkit_LIBS: ${CUDAToolkit_LIBS}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

#set opencv
message("CMAKE_FIND_USE_PACKAGE_ROOT_PATH: ${CMAKE_FIND_USE_PACKAGE_ROOT_PATH}")
set(CMAKE_FIND_USE_PACKAGE_ROOT_PATH TRUE)
set(CMAKE_PREFIX_PATH "/home/br/program/opencv3/build"
                      "${CMAKE_PREFIX_PATH}")
set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} "conv_test.cu")
target_include_directories(${PROJECT_NAME} PUBLIC
                        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC 
                      ${CUDAToolkit_LIBS}
                      ${OpenCV_LIBS})