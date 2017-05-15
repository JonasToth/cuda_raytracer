
enable_testing()

# google test as testing framework must be available
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# unit tests
add_executable(test_obj_io.x test_obj_io.cpp)
target_link_libraries(test_obj_io.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

add_executable(test_glfw.x test_glfw.cpp)
target_link_libraries(test_glfw.x ${GTEST_MAIN} ${GTEST_LIBRARIES} glfw ${OPENGL_gl_LIBRARY} pthread)

#add_executable(test_ad_san.x test_ad_san.cpp)
#target_link_libraries(test_ad_san.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

cuda_add_executable(test_build_cuda.x test_build_cuda.cu ${cudafiles})
target_link_libraries(test_build_cuda.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)


# ----------------------------------------------------------------------------------------
#                               ===== Testing =====

add_test(test_obj_io test_obj_io.x)

#add_test(test_ad_san test_ad_san.x)
#set_tests_properties(test_ad_san PROPERTIES 
#                     PASS_REGULAR_EXPRESSION "AddressSanitizer: stack-buffer-overflow")

add_test(test_build_cuda test_build_cuda.x)
#set_tests_properties(test_build_cuda)
#PROPERTIES
#PASS_REGULAR_EXPRESSION "LeakSanitizer: detected memory leaks")

# probably not a term with X enabled in the buildbot, but manually running it works
add_test(test_glfw test_glfw.x)
set_tests_properties(test_glfw PROPERTIES PASS_REGULAR_EXPESSION "Test failed")
