enable_testing()

# google test as testing framework must be available
find_package(GTest REQUIRED) 
include_directories(${GTEST_INCLUDE_DIRS}) 


# symbolic links to all test files
add_custom_target(example_files
    # necessary for integration tests
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/validate.sh ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_cube.png ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_materials.png ${PROJECT_BINARY_DIR}/test

    # object files and materials necessary as testcases
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.mtl ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_no_normals.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_no_normals.mtl ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_no_normals_no_materials.obj ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/shapes.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/bad.obj ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.mtl ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.mtl ${PROJECT_BINARY_DIR}/test
)


# ======================== DEMOING ===================================


cuda_add_executable(demo_cuda_draw_buffer.x demo_cuda_draw_buffer.cu
                    ${PROJECT_SOURCE_DIR}/src/obj_io.cu
                    ${PROJECT_SOURCE_DIR}/src/management/input_manager.cpp
                    ${PROJECT_SOURCE_DIR}/src/management/input_callback.cpp
                    ${PROJECT_SOURCE_DIR}/src/management/surface_raii.cu 
                    ${PROJECT_SOURCE_DIR}/src/management/window.cpp)
target_link_libraries(demo_cuda_draw_buffer.x 
                      ${GTEST_MAIN} ${GTEST_LIBRARIES} 
                      pthread 
                      glfw ${OPENGL_gl_LIBRARY}
                      ${PNG_LIBRARIES})
add_dependencies(demo_cuda_draw_buffer.x example_files)


cuda_add_executable(demo_render_materials.x demo_render_materials.cu
                    ${PROJECT_SOURCE_DIR}/src/obj_io.cu
                    ${PROJECT_SOURCE_DIR}/src/management/input_manager.cpp
                    ${PROJECT_SOURCE_DIR}/src/management/input_callback.cpp
                    ${PROJECT_SOURCE_DIR}/src/management/surface_raii.cu 
                    ${PROJECT_SOURCE_DIR}/src/management/window.cpp)
target_link_libraries(demo_render_materials.x 
    ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread
    glfw ${OPENGL_gl_LIBRARY}
    ${PNG_LIBRARIES})
add_dependencies(demo_render_materials.x example_files)


cuda_add_executable(demo_draw_mini.x demo_draw_mini.cu
                    ${PROJECT_SOURCE_DIR}/src/obj_io.cu
                    ${PROJECT_SOURCE_DIR}/src/management/input_manager.cpp
                    ${PROJECT_SOURCE_DIR}/src/management/input_callback.cpp
                    ${PROJECT_SOURCE_DIR}/src/management/surface_raii.cu 
                    ${PROJECT_SOURCE_DIR}/src/management/window.cpp)
target_link_libraries(demo_draw_mini.x 
                      ${GTEST_MAIN} ${GTEST_LIBRARIES} 
                      pthread 
                      glfw ${OPENGL_gl_LIBRARY}
                      ${PNG_LIBRARIES})
add_dependencies(demo_draw_mini.x example_files)



# ======================== TESTING ===================================


# unit tests
cuda_add_executable(test_build_cuda.x test_build_cuda.cu)
target_link_libraries(test_build_cuda.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_camera.x test_camera.cpp)
target_link_libraries(test_camera.x ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_camera_control.x test_camera_control.cpp
                ${PROJECT_SOURCE_DIR}/src/management/input_manager.cpp)
target_link_libraries(test_camera_control.x ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_glfw.x test_glfw.cpp
               ${PROJECT_SOURCE_DIR}/src/management/window.cpp)
target_link_libraries(test_glfw.x ${GTEST_MAIN} ${GTEST_LIBRARIES} glfw ${OPENGL_gl_LIBRARY} pthread)


add_executable(test_input_manager.x test_input_manager.cpp
               ${PROJECT_SOURCE_DIR}/src/management/input_manager.cpp)
target_link_libraries(test_input_manager.x ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_light.x test_light.cpp)
target_link_libraries(test_light.x ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_material.x test_material.cpp)
target_link_libraries(test_material.x ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)


cuda_add_executable(test_obj_io.x test_obj_io.cu 
                    ${PROJECT_SOURCE_DIR}/src/obj_io.cu)
target_link_libraries(test_obj_io.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)
add_dependencies(test_obj_io.x example_files)


add_executable(test_ray.x test_ray.cpp)
target_link_libraries(test_ray.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)


cuda_add_executable(test_surface_raii.x test_surface_raii.cu
                    ${PROJECT_SOURCE_DIR}/src/management/surface_raii.cu)
target_link_libraries(test_surface_raii.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread glfw ${OPENGL_gl_LIBRARY} ${PNG_LIBRARIES})


add_executable(test_triangle.x test_triangle.cpp)
target_link_libraries(test_triangle.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_shading.x test_shading.cpp)
target_link_libraries(test_shading.x 
                      ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_executable(test_vector.x test_vector.cpp)
target_link_libraries(test_vector.x ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)


add_test(unit_build_cuda test_build_cuda.x)
add_test(unit_camera test_camera.x)
add_test(unit_camera_control test_camera_control.x)
add_test(unit_input_manager test_input_manager.x)
add_test(unit_light test_light.x)
add_test(unit_obj_io test_obj_io.x)
add_test(unit_material test_material.x)
add_test(unit_ray test_ray.x)
add_test(unit_shading test_shading.x)
add_test(unit_surface_raii test_surface_raii.x)
add_test(unit_triangle test_triangle.x)
add_test(unit_vector test_vector.x)

# ====================== Integration tests for "real world" scenarios ====================

cuda_add_executable(int_test_cube.x int_test_cube.cu
                    ${PROJECT_SOURCE_DIR}/src/obj_io.cu
                    ${PROJECT_SOURCE_DIR}/src/management/surface_raii.cu
                    ${PROJECT_SOURCE_DIR}/src/management/window.cpp)
target_link_libraries(int_test_cube.x 
                      glfw ${OPENGL_gl_LIBRARY}
                      ${PNG_LIBRARIES})

cuda_add_executable(int_test_materials.x int_test_materials.cu
                    ${PROJECT_SOURCE_DIR}/src/obj_io.cu
                    ${PROJECT_SOURCE_DIR}/src/management/surface_raii.cu
                    ${PROJECT_SOURCE_DIR}/src/management/window.cpp)
target_link_libraries(int_test_materials.x 
                      glfw ${OPENGL_gl_LIBRARY}
                      ${PNG_LIBRARIES})


add_test(NAME integration_cube 
         COMMAND ./validate.sh ./int_test_cube.x cube.png reference_cube.png)
add_test(NAME integration_materials
         COMMAND ./validate.sh ./int_test_materials.x materials.png reference_materials.png)
