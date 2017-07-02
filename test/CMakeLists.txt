enable_testing()

add_custom_target(example_files
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/shapes.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.mtl ${PROJECT_BINARY_DIR}/test
)

# google test as testing framework must be available
find_package(GTest REQUIRED) 
include_directories(${GTEST_INCLUDE_DIRS}) 
# unit tests
cuda_add_executable(test_obj_io.x test_obj_io.cu ${PROJECT_SOURCE_DIR}/src/obj_io.cu)
target_link_libraries(test_obj_io.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)
add_dependencies(test_obj_io.x example_files)

add_executable(test_vector.x test_vector.cpp)
target_link_libraries(test_vector.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

cuda_add_executable(test_cuda_vector.x test_cuda_vector.cu)
target_link_libraries(test_cuda_vector.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

add_executable(test_triangle.x test_triangle.cpp)
target_link_libraries(test_triangle.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

cuda_add_executable(test_cuda_triangle.x test_cuda_triangle.cu)
target_link_libraries(test_cuda_triangle.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

add_executable(test_ray.x test_ray.cpp)
target_link_libraries(test_ray.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

cuda_add_executable(test_cuda_ray.x test_cuda_ray.cu)
target_link_libraries(test_cuda_ray.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)

add_executable(test_glfw.x test_glfw.cpp)
target_link_libraries(test_glfw.x ${GTEST_MAIN} ${GTEST_LIBRARIES} glfw ${OPENGL_gl_LIBRARY} pthread)

cuda_add_executable(test_cuda_draw_buffer.x test_cuda_draw_buffer.cu)
target_link_libraries(test_cuda_draw_buffer.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread glfw ${OPENGL_gl_LIBRARY})

cuda_add_executable(test_build_cuda.x test_build_cuda.cu ${cudafiles})
target_link_libraries(test_build_cuda.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)


# ----------------------------------------------------------------------------------------
#                               ===== Testing =====

add_test(test_obj_io test_obj_io.x)
add_test(test_build_cuda test_build_cuda.x)
add_test(test_cuda_vector test_cuda_vector.x)
add_test(test_vector test_vector.x)
add_test(test_cuda_triangle test_cuda_triangle.x)
add_test(test_triangle test_triangle.x)
add_test(test_cuda_ray test_cuda_ray.x)
add_test(test_ray test_ray.x)

#set_tests_properties(test_build_cuda)
#PROPERTIES
#PASS_REGULAR_EXPRESSION "LeakSanitizer: detected memory leaks")

# probably not a term with X enabled in the buildbot, but manually running it works
# add_test(test_glfw test_glfw.x)
# set_tests_properties(test_glfw PROPERTIES PASS_REGULAR_EXPESSION "FAILED")
