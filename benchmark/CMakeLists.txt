# configure GBench
set(BENCHMARK_ENABLE_TESTING OFF)
include_directories("${googlebenchmark_SOURCE_DIR}/include")

# testing will be used to run all benchmarks
enable_testing()

add_executable(bench_dummy.x bench_dummy.cpp)
target_link_libraries(bench_dummy.x benchmark pthread)



add_test(bench_dummy bench_dummy.x)
