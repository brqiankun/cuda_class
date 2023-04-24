cmake_minimum_required(VERSION 3.20)
project(br_test_cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(CUDAToolkit_ROOT "/home/rui.bai/cuda-11.3")
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
    message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
else()
    message(FATAL_ERROR "project: ${PROJECT_NAME} not found cuda")
endif()

file(GLOB_RECURSE CPP_FILES src/*.cpp src/*cu)
message(STATUS "CPP_FILES : ${CPP_FILES}")

# add_executable(${PROJECT_NAME} src/asyncAPI_test.cu)
add_executable(${PROJECT_NAME} src/simpleZeroCopy_br.cu)
target_include_directories(${PROJECT_NAME} PUBLIC include)
# target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
if (CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
endif()







