cmake_minimum_required(VERSION 3.0)
project(onnx2trt)

enable_language(CUDA CXX) #激活CUDA语言支持，使用第一个写法时要进行注释

#设置语言标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_PREFIX_PATH "/home/rui.bai/download/libtorch")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if (NOT Torch_FOUND)
    message (FATAL_ERROR "Libtorch Not Found!")
endif (NOT Torch_FOUND)
message (STATUS "Libtorch status: ")
message (STATUS "libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch_FOUND ${Torch_FOUND}")
message(STATUS "Torch_DIR ${Torch_DIR}")

set(CUDAToolkit_ROOT "/home/rui.bai/cuda-11.3")
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_FOUND ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_DIR ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDAToolkit_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIR}")

set(TENSORRT_LIB_PATH "/home/rui.bai/download/TensorRT-8.5.3.1/lib")
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so*")
file(GLOB CUDAToolkit_LIBS "${CUDAToolkit_LIBRARY_DIR}/*.so*")

add_executable(${PROJECT_NAME} onnx2trt_engine.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
                           ${CUDAToolkit_INCLUDE_DIRS}
                           "/home/rui.bai/download/TensorRT-8.5.3.1/include"
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