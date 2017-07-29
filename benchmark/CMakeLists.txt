# configure GBench
include_directories("${googlebenchmark_SOURCE_DIR}/include")

# testing will be used to run all benchmarks
enable_testing()

# files for benchmarking
add_custom_target(benchmark_files
    # specialized cases
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_1.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_1.mtl ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_2.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_2.mtl ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_3.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_3.mtl ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_4.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_4.mtl ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_5.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_5.mtl ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_6.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/benchmark/files/cube_subdiv_6.mtl ${PROJECT_BINARY_DIR}/benchmark

    # real world
    #COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.obj ${PROJECT_BINARY_DIR}/benchmark
    #COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.mtl ${PROJECT_BINARY_DIR}/benchmark
)

# benchmarks

cuda_add_executable(bench_basic_scene.x bench_basic_scene.cu)
target_link_libraries(bench_basic_scene.x 
                      management
                      benchmark pthread glfw ${OPENGL_gl_LIBRARY} ${PNG_LIBRARIES})
add_dependencies(bench_basic_scene.x benchmark_files)


# benchmarks mostly for traversing geometry
cuda_add_executable(bench_cube_subdivision.x bench_cube_subdivision.cu)
target_link_libraries(bench_cube_subdivision.x 
                      management
                      benchmark pthread glfw ${OPENGL_gl_LIBRARY} ${PNG_LIBRARIES})
add_dependencies(bench_cube_subdivision.x benchmark_files)

add_executable(bench_cube_subdivision_cpu.x bench_cube_subdivision.cpp
               ${PROJECT_SOURCE_DIR}/src/management/world.cu)
target_compile_options(bench_cube_subdivision_cpu.x PUBLIC "-fopenmp")
target_compile_definitions(bench_cube_subdivision_cpu.x
                           PRIVATE -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
target_link_libraries(bench_cube_subdivision_cpu.x 
                      cpu_kernels pthread
                      benchmark ${PNG_LIBRARIES})
add_dependencies(bench_cube_subdivision_cpu.x benchmark_files)



#add_test(NAME bench_basic_scene 
#         COMMAND ./bench_basic_scene.x --benchmark_color=false --benchmark_counters_tabular=true)
add_test(NAME bench_cube_subdivision 
         COMMAND ./bench_cube_subdivision.x --benchmark_color=false --benchmark_counters_tabular=true)
add_test(NAME bench_cube_subdivision_cpu
         COMMAND ./bench_cube_subdivision_cpu.x --benchmark_color=false --benchmark_counters_tabular=true)
