cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(test_libtorch)

set(CMAKE_PREFIX_PATH "/home/br/program/libtorch"
                      ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_libtorch test_libtorch.cpp)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}")
set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 14)
