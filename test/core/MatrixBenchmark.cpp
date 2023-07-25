#include <benchmark/benchmark.h>
#include "../../src/core/Matrix.hpp"

static void LargeMatrixInitialization(benchmark::State& state) {
    for (auto _ : state) {
        Matrix<int> m(10000, 10000);
    }
}

BENCHMARK(LargeMatrixInitialization);