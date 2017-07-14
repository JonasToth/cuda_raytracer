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
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_materials_smooth.png ${PROJECT_BINARY_DIR}/test

    # object files and materials necessary as testcases
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_subdiv_1.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_subdiv_1.mtl ${PROJECT_BINARY_DIR}/test
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

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene_smooth.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene_smooth.mtl ${PROJECT_BINARY_DIR}/test
)


# ======================== DEMOING ===================================

SET(DEMOS
    cuda_draw_buffer
    render_materials
    draw_mini
    )


FOREACH(demo IN LISTS DEMOS)
    cuda_add_executable(demo_${demo}.x demo_${demo}.cu)
    target_link_libraries(demo_${demo}.x 
                          management
                          ${GTEST_MAIN} ${GTEST_LIBRARIES} 
                          pthread 
                          glfw ${OPENGL_gl_LIBRARY}
                          ${PNG_LIBRARIES})
    add_dependencies(demo_${demo}.x example_files)
ENDFOREACH(demo)


# =============================== UNIT TESTING ===================================

SET(UNITTESTS_CUDA
    build_cuda
    world
    surface_raii
    )

FOREACH(cuda_unit IN LISTS UNITTESTS_CUDA)
    cuda_add_executable(test_${cuda_unit}.x test_${cuda_unit}.cu)
    target_link_libraries(test_${cuda_unit}.x 
                          management
                          ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread)
    add_test(unit_${cuda_unit} test_${cuda_unit}.x)
ENDFOREACH(cuda_unit)

# additional stuff, that is not done for all tests, but for some
target_link_libraries(test_surface_raii.x 
                      ${GTEST_MAIN} ${GTEST_LIBRARIES} pthread 
                      glfw ${OPENGL_gl_LIBRARY} ${PNG_LIBRARIES})
add_dependencies(test_world.x example_files)


# unit tests, that are plain c++ are more isolated
SET(UNITTESTS_PLAIN
    camera
    camera_control
    glfw
    input_manager
    light
    material
    ray
    triangle
    shading
    vector
    )

FOREACH(unit_test IN LISTS UNITTESTS_PLAIN)
    add_executable(test_${unit_test}.x test_${unit_test}.cpp)
    target_link_libraries(test_${unit_test}.x 
                          management
                          glfw ${OPENGL_gl_LIBRARY}
                          ${GEST_MAIN} ${GTEST_LIBRARIES} pthread)
    add_test(unit_${unit_test} test_${unit_test}.x)
ENDFOREACH(unit_test)

# ====================== Integration tests for "real world" scenarios ====================

SET(INTEGRATION_TESTS
    cube
    cube_smooth
    materials
    materials_smooth
    )
    

FOREACH(test IN LISTS INTEGRATION_TESTS)
    cuda_add_executable(int_test_${test}.x int_test_${test}.cu)
    target_link_libraries(int_test_${test}.x
                          management
                          utility
                          glfw ${OPENGL_gl_LIBRARY}
                          ${PNG_LIBRARIES})

    add_test(NAME integration_${test}
             COMMAND ./validate.sh ./int_test_${test}.x ${test}.png reference_${test}.png)
ENDFOREACH(test)
