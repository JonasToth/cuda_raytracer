# configure GBench
include_directories("${googlebenchmark_SOURCE_DIR}/include")

# testing will be used to run all benchmarks
enable_testing()

# files for benchmarking
add_custom_target(benchmark_files
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/cube.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.obj ${PROJECT_BINARY_DIR}/benchmark
    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/examples/material_scene.mtl ${PROJECT_BINARY_DIR}/benchmark
)

# benchmarks

add_executable(bench_dummy.x bench_dummy.cpp)
target_link_libraries(bench_dummy.x benchmark pthread)

add_executable(bench_basic_scene.x bench_basic_scene.cu)
target_link_libraries(bench_basic_scene.x 
    benchmark pthread glfw ${OPENGL_gl_LIBRARY} ${PNG_LIBRARIES})
add_dependencies(bench_basic_scene.x benchmark_files)


add_test(bench_dummy bench_dummy.x)
add_test(bench_basic_scene bench_basic_scene.x)
