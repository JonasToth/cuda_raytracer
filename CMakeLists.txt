# cmake file to build the project and tests
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# ----------------------------------------------------------------------------------------
#                              ===== Project Setup =====
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
# find libraries
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# cuda

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====

# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====
# executable raytracer
add_executable(raytracer.x src/main.cpp)

# unit tests
add_executable(test_obj_io.x test/test_obj_io.cpp)
target_link_libraries(test_obj_io.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

# ----------------------------------------------------------------------------------------
#                               ===== Testing =====
enable_testing()
add_test(test_obj_io test_obj_io.x)
