# cmake file to build the project and tests
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# ----------------------------------------------------------------------------------------
#                              ===== Project Setup =====
project(CUDA_Raytracer)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)

# cmake modules
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )
# guard against in source builds and bad build-type strings
include(safeguards)

# cuda
find_package(CUDA REQUIRED)
# opengl and libraries to make windows and stuff
find_package(glfw3 3.2 REQUIRED) # http://www.glfw.org/docs/latest/build_guide.html#build_link_cmake_package
find_package(OpenGL REQUIRED)


# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====
add_definitions("--std=c++11 -Wall -pedantic")
include_directories(src)
include_directories(lib)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 -O3 -Wno-deprecated-gpu-targets --expt-extended-lambda -DTHRUST_DEBUG")

# optimize in release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# build debug with sanitizer
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_CXX_EXE_LINKER_FLAGS_DEBUG "${CMAKE_CXX_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")

if(${COMPILER_ID} MATCHES "GNU") # gcc specific flags
    add_definitions("-Wextra -Wpedantic")
elseif(${COMPILER_ID} MATCHES "clang") # clang specific flags
    add_definitions("-Weverything")
endif()

# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====

# executable raytracer
cuda_add_executable(raytracer.x src/main.cpp)


add_subdirectory(${PROJECT_SOURCE_DIR}/lib/googletest/googletest)
add_subdirectory(test)

add_subdirectory(benchmark)
