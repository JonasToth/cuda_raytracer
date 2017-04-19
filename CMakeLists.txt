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
find_package(CUDA REQUIRED)

# ----------------------------------------------------------------------------------------
#                         ===== Compiler Configuration =====
if(NOT UNIX)
    message(FATAL_ERROR "Unix system required!")
endif()

add_definitions("--std=c++14")

set(${CMAKE_CXX_FLAGS_RELEASE} "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(${CMAKE_CXX_FLAGS} "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(${CMAKE_CXX_LINK_EXECUTABLE} "${CMAKE_CXX_LINK_EXECUTABLE} -fsanitize=address")
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)


# ----------------------------------------------------------------------------------------
#                              ===== Build targets =====
# executable raytracer
add_executable(raytracer.x src/main.cpp)

# unit tests
add_executable(test_obj_io.x test/test_obj_io.cpp)
target_link_libraries(test_obj_io.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

add_executable(test_ad_san.x test/test_ad_san.cpp)
target_link_libraries(test_ad_san.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)
# ----------------------------------------------------------------------------------------
#                               ===== Testing =====
enable_testing()

add_test(test_obj_io test_obj_io.x)

add_test(test_ad_san test_ad_san.x)
set_tests_properties(test_ad_san PROPERTIES 
                     PASS_REGULAR_EXPRESSION "AddressSanitizer: stack-buffer-overflow")

