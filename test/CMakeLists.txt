enable_testing()

# google test as testing framework must be available
find_package(GTest REQUIRED) 
include_directories(${GTEST_INCLUDE_DIRS}) 


# ======================== DEMOING ===================================

add_custom_target(demo_files
    # infrastructure
    #COMMAND mkdir -p ${PROJECT_BINARY_DIR}/test/demo_input/

    # demo files
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_flat.obj ${PROJECT_BINARY_DIR}/test/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_flat.mtl ${PROJECT_BINARY_DIR}/test/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_smooth.obj ${PROJECT_BINARY_DIR}/test/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_smooth.mtl ${PROJECT_BINARY_DIR}/test/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.obj ${PROJECT_BINARY_DIR}/test/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.mtl ${PROJECT_BINARY_DIR}/test/

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_reduced.obj ${PROJECT_BINARY_DIR}/test/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_reduced.mtl ${PROJECT_BINARY_DIR}/test/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/complex.obj ${PROJECT_BINARY_DIR}/test/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/complex.mtl ${PROJECT_BINARY_DIR}/test/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/structured_underground.obj ${PROJECT_BINARY_DIR}/test/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/structured_underground.mtl ${PROJECT_BINARY_DIR}/test/ 
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
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_flat.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_flat.mtl ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_subdiv_1.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_subdiv_1.mtl ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_no_normals.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_no_normals.mtl ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_no_normals_no_materials.obj ${PROJECT_BINARY_DIR}/test


    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/shapes.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/bad.obj ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/mini_cooper.mtl ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_flat.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_flat.mtl ${PROJECT_BINARY_DIR}/test

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_smooth.obj ${PROJECT_BINARY_DIR}/test
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_smooth.mtl ${PROJECT_BINARY_DIR}/test
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
    COMMAND mkdir -p ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/validate.sh ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/compare_images.py ${PROJECT_BINARY_DIR}/test/integration_tests/

    # actual test files
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_flat.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_flat.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_smooth.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_smooth.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_room_flat.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube_room_flat.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/ 

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_flat.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_flat.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_smooth.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/materials_smooth.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/structured_underground.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/structured_underground.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/

    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/complex.obj ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/complex.mtl ${PROJECT_BINARY_DIR}/test/integration_tests/

    # reference as ground truth
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_flat_cube.png ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_smooth_cube.png ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_flat_materials.png ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_smooth_materials.png ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_flat_cube_room.png ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_structured_underground.png ${PROJECT_BINARY_DIR}/test/integration_tests/
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/test/reference/reference_complex.png ${PROJECT_BINARY_DIR}/test/integration_tests/
)


# ------------------------------- All renderers -----------------------------------------
SET(CPU_RENDER
    flat_phong_shader
    smooth_phong_shader
    )
SET(GPU_RENDER
    flat_phong_shader
    smooth_phong_shader
    )

FOREACH(renderer IN LISTS CPU_RENDER)
    add_executable(cpu_${renderer}.x ${renderer}.cpp
                   ${PROJECT_SOURCE_DIR}/src/management/world.cu)
    target_compile_options(cpu_${renderer}.x PUBLIC "-fopenmp")
    target_compile_definitions(cpu_${renderer}.x 
                               PRIVATE -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    target_link_libraries(cpu_${renderer}.x
                          cpu_kernels pthread
                          ${PNG_LIBRARIES})
    add_dependencies(cpu_${renderer}.x integration_test_files)
ENDFOREACH()


FOREACH(renderer IN LISTS GPU_RENDER)
    cuda_add_executable(gpu_${renderer}.x ${renderer}.cu)
    add_dependencies(gpu_${renderer}.x integration_test_files)
    target_link_libraries(gpu_${renderer}.x
                          management 
                          glfw ${OPENGL_gl_LIBRARY}
                          ${PNG_LIBRARIES})
    add_dependencies(gpu_${renderer}.x integration_test_files)
ENDFOREACH()


# --------------- Running these renderes with different input files ---------------------
SET(FLAT_INTEGRATION_TESTS
    cube
    cube_room
    materials
    )
SET(SMOOTH_INTEGRATION_TESTS
    cube
    materials
    )

FOREACH(test IN LISTS FLAT_INTEGRATION_TESTS)
    add_test(NAME gpu_flat_${test}
             COMMAND ./validate.sh ../gpu_flat_phong_shader.x ${test}_flat.obj gpu_flat_${test}.png reference_flat_${test}.png
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/integration_tests)
    add_test(NAME cpu_flat_${test}
             COMMAND ./validate.sh ../cpu_flat_phong_shader.x ${test}_flat.obj cpu_flat_${test}.png reference_flat_${test}.png
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/integration_tests)
ENDFOREACH(test)

FOREACH(test IN LISTS SMOOTH_INTEGRATION_TESTS)
    add_test(NAME gpu_smooth_${test}
             COMMAND ./validate.sh ../gpu_smooth_phong_shader.x ${test}_smooth.obj gpu_smooth_${test}.png reference_smooth_${test}.png
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/integration_tests)
    add_test(NAME cpu_smooth_${test}
             COMMAND ./validate.sh ../cpu_smooth_phong_shader.x ${test}_smooth.obj cpu_smooth_${test}.png reference_smooth_${test}.png
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/integration_tests)
ENDFOREACH(test)

add_test(NAME scene_structured_underground
         COMMAND ./validate.sh ../gpu_smooth_phong_shader.x structured_underground.obj structured_underground.png reference_structured_underground.png
         WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/integration_tests)
add_test(NAME scene_complex
         COMMAND ./validate.sh ../gpu_smooth_phong_shader.x complex.obj complex.png reference_complex.png
         WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test/integration_tests)
