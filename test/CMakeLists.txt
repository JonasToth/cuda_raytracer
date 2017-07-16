enable_testing()

# google test as testing framework must be available
find_package(GTest REQUIRED) 
include_directories(${GTEST_INCLUDE_DIRS}) 


# ======================== DEMOING ===================================

add_custom_target(demo_files
    # infrastructure
    COMMAND mkdir -p ${PROJECT_BINARY_DIR}/test/demo_input/

    # demo files
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.obj ${PROJECT_BINARY_DIR}/test/demo_input/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.mtl ${PROJECT_BINARY_DIR}/test/ # stupid obj will look in exectuin directory

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.obj ${PROJECT_BINARY_DIR}/test/demo_input/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.mtl ${PROJECT_BINARY_DIR}/test/ # stupid obj will look in exectuin directory

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_reduced.obj ${PROJECT_BINARY_DIR}/test/demo_input/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_reduced.mtl ${PROJECT_BINARY_DIR}/test/ # stupid obj will look in exectuin directory
)


SET(DEMOS
    cuda_draw_buffer
    render_materials
    draw_mini
    )


FOREACH(demo IN LISTS DEMOS)
    cuda_add_executable(demo_${demo}.x demo_${demo}.cu)
    target_link_libraries(demo_${demo}.x 
                          management cpu_kernels
                          ${GTEST_MAIN} ${GTEST_LIBRARIES} 
                          pthread 
                          glfw ${OPENGL_gl_LIBRARY}
                          ${PNG_LIBRARIES})
    add_dependencies(demo_${demo}.x demo_files)
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

# symbolic links to all test files
add_custom_target(example_files
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


add_dependencies(test_world.x example_files)


# unit tests, that are plain c++ are more isolated
SET(UNITTESTS_PLAIN
    camera
    camera_control
    glfw
    input_manager
    #kernel_from_cpp
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

add_executable(test_kernel_from_cpp.x test_kernel_from_cpp.cpp
               ${PROJECT_SOURCE_DIR}/src/graphic/kernels/trace.cpp
               ${PROJECT_SOURCE_DIR}/src/graphic/kernels/utility.cpp)
target_link_libraries(test_kernel_from_cpp.x 
                      management
                      ${GEST_MAIN} ${GTEST_LIBRARIES} pthread
                      ${PNG_LIBRARIES})
add_test(unit_test_kernel_from_cpp test_kernel_from_cpp.x)

# ====================== Integration tests for "real world" scenarios ====================
add_custom_target(integration_test_files
    # necessary for integration tests
    
    # infrastructure
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/validate.sh ${PROJECT_BINARY_DIR}/test
    COMMAND mkdir -p ${PROJECT_BINARY_DIR}/test/int_test_output/
    COMMAND mkdir -p ${PROJECT_BINARY_DIR}/test/int_test_input/
    COMMAND mkdir -p ${PROJECT_BINARY_DIR}/test/int_test_ref/

    # actual test files
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.obj ${PROJECT_BINARY_DIR}/test/int_test_input/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.mtl ${PROJECT_BINARY_DIR}/test/int_test_input/ # stupid obj load will look in exectuion dir
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.mtl ${PROJECT_BINARY_DIR}/test/ # stupid obj load will look in exectuion dir

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_smooth.obj ${PROJECT_BINARY_DIR}/test/int_test_input/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_smooth.mtl ${PROJECT_BINARY_DIR}/test/int_test_input/ # stupid obj load will look in exectuion dir
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_smooth.mtl ${PROJECT_BINARY_DIR}/test/ # stupid obj load will look in exectuion dir

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.obj ${PROJECT_BINARY_DIR}/test/int_test_input/materials.obj
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.mtl ${PROJECT_BINARY_DIR}/test/int_test_input/materials.mtl # stupid obj load will look in exectuion dir
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.mtl ${PROJECT_BINARY_DIR}/test/materials.mtl # stupid obj load will look in exectuion dir

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene_smooth.obj ${PROJECT_BINARY_DIR}/test/int_test_input/materials_smooth.obj
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene_smooth.mtl ${PROJECT_BINARY_DIR}/test/int_test_input/materials_smooth.mtl # stupid obj load will look in exectuion dir
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene_smooth.mtl ${PROJECT_BINARY_DIR}/test/materials_smooth.mtl # stupid obj load will look in exectuion dir

    # reference as ground truth
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_cube.png ${PROJECT_BINARY_DIR}/test/int_test_ref/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_cube_smooth.png ${PROJECT_BINARY_DIR}/test/int_test_ref/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_materials.png ${PROJECT_BINARY_DIR}/test/int_test_ref/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_materials_smooth.png ${PROJECT_BINARY_DIR}/test/int_test_ref/
)


SET(INTEGRATION_TESTS
    cube
    cube_smooth
    materials
    materials_smooth
    )
    

FOREACH(test IN LISTS INTEGRATION_TESTS)
    # run this kernel and image on GPU AND CPU
    cuda_add_executable(int_test_gpu_${test}.x int_test_${test}.cu)
    target_link_libraries(int_test_gpu_${test}.x
                          management cpu_kernels
                          glfw ${OPENGL_gl_LIBRARY}
                          ${PNG_LIBRARIES})
    add_dependencies(int_test_gpu_${test}.x integration_test_files)

    add_test(NAME integration_gpu_${test}
             COMMAND ./validate.sh ./int_test_gpu_${test}.x ${test} gpu_${test} reference_${test})

    # CPU executable
    add_executable(int_test_cpu_${test}.x 
                   int_test_${test}.cpp
                   ${PROJECT_SOURCE_DIR}/src/management/world.cu)
    target_compile_options(int_test_cpu_${test}.x PUBLIC "-fopenmp")
    #target_link_options(int_test_cpu_${test}.x PRIVATE "-fopenmp")
    target_compile_definitions(int_test_cpu_${test}.x 
                               PRIVATE -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    target_link_libraries(int_test_cpu_${test}.x
                          cpu_kernels pthread
                          glfw ${OPENGL_gl_LIBRARY}
                          ${PNG_LIBRARIES})
    add_dependencies(int_test_cpu_${test}.x integration_test_files)

    add_test(NAME integration_cpu_${test}
             COMMAND ./validate.sh ./int_test_cpu_${test}.x ${test} cpu_${test} reference_${test})
ENDFOREACH(test)
