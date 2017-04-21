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
# find libraries
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# cuda
find_package(CUDA REQUIRED)

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====
# optimize in release

add_definitions("--std=c++14 -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# build debug with sanitizer
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_EXE_LINKER_FLAGS_DEBUG "${CMAKE_CXX_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")

file(GLOB cudafiles *.cu)


if(${COMPILER_ID} MATCHES "GNU") # gcc specific flags

elseif(${COMPILER_ID} MATCHES "clang") # clang specific flags

endif()


# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====
# executable raytracer
add_executable(raytracer.x src/main.cpp)

if(${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
    add_subdirectory(test)
endif()
