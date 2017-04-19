# cmake file to build the project and tests
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(CUDA_Raytracer)
enable_language(C CXX)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)

# cmake modules
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )

# guard against in source builds and bad build-type strings
include(safeguards)

# executable raytracer
add_executable(raytracer.x src/main.cpp)


# TODO testing must be added later
