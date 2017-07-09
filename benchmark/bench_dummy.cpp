#include "benchmark/benchmark_api.h"
#include <vector>

static void BM_VectorInsert(benchmark::State& state)
{
    while(state.KeepRunning())
    {
        std::vector<int> insertion_test;
        for(int i = 0, i_end = state.range_x(); i < i_end; ++i)
            insertion_test.push_back(i);
    }
}

BENCHMARK(BM_VectorInsert)->Range(8,8 << 10);

BENCHMARK_MAIN();
