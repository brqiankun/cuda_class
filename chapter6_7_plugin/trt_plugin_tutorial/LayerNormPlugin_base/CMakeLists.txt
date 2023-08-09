cmake_minimum_required(VERSION 3.20)
project(test_LayerNorm_kernel CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(CUDAToolkit_ROOT "/usr/local/cuda-11.3")
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit_FOUND ${CUDAToolkit_FOUND}")
    message(STATUS "CUDAToolkit_DIR ${CUDAToolkit_BIN_DIR}")
    message(STATUS "CUDAToolkit_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "project: ${PROJECT_NAME} not found cuda")
endif()

file(GLOB_RECURSE CPP_FILES src/*.cpp src/*cu)
message(STATUS "CPP_FILES : ${CPP_FILES}")

set(TENSORRT_INCLUDE_DIR "/home/br/program/TensorRT-8.5.3.1/include")
set(TENSORRT_LIB_PATH "/home/br/program/TensorRT-8.5.3.1/lib")
file(GLOB TRT_LIBS "${TENSORRT_LIB_PATH}/*.so*")

# add_executable(${PROJECT_NAME} src/asyncAPI_test.cu)
add_executable(${PROJECT_NAME} "LayerNormPlugin.cu" "test_LayerNorm_kernel.cu")
target_include_directories(${PROJECT_NAME} PUBLIC
                           ${CUDAToolkit_INCLUDE_DIRS}
                           ${TENSORRT_INCLUDE_DIR}
                           ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} 
                      ${CUDAToolkit_LIBS}
                      ${TRT_LIBS})

# target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
if (CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
endif()