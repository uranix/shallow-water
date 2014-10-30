cmake_minimum_required(VERSION 2.8)

project(shallow)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-Wno-invalid-offsetof" COMPILER_SUPPORTS_INVALID_OFFSETOF)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
	message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
if(COMPILER_SUPPORTS_INVALID_OFFSETOF)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

find_package(CUDA REQUIRED)

add_executable(shallow shallow.cpp)
include_directories(cuda-helper/ ${CUDA_INCLUDE_DIRS})
target_link_libraries(shallow ${CUDA_CUDA_LIBRARY})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -arch=sm_12")
set(CUDA_HOST_COMPILER g++)
CUDA_COMPILE_PTX(kernels kernels.cu)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/kernels.ptx COMMAND ${CMAKE_COMMAND} -E copy ${kernels} ${CMAKE_BINARY_DIR}/kernels.ptx DEPENDS ${kernels})
add_custom_target(ptx ALL DEPENDS ${CMAKE_BINARY_DIR}/kernels.ptx kernels.cu SOURCES kernels.cu)
