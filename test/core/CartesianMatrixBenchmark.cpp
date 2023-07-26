#include <benchmark/benchmark.h>
#include "../../src/core/CartesianMatrix.hpp"

static void largeCartesianMatrixInitializationBenchmark(benchmark::State& state) {
    for (auto _ : state) {
        CartesianMatrix<float> m(10000, 10000);
    }
}
BENCHMARK(largeCartesianMatrixInitializationBenchmark);

static void largeCartesianMatrixArrayInitializationBenchmark(benchmark::State& state) {
    CartesianMatrix<float> m[9];
    for (int i = 0; i < 9; ++i) { 
        m[i] = CartesianMatrix<float>(10000, 10000);
    }
    for (auto _ : state) {
#pragma omp parallel for 
        for(int i = 0; i < 9; ++i) {
            m[i].fill(0.0F);
        }
    }
}
BENCHMARK(largeCartesianMatrixArrayInitializationBenchmark);

static void batchRevisionXBenchmark(benchmark::State& state) {
    CartesianMatrix<float> m(10000, 10000);
    for (auto _ : state) {
        m.batchRevisionX(2532, 6);
    }
}
BENCHMARK(batchRevisionXBenchmark);

static void batchRevisionYBenchmark(benchmark::State& state) {
    CartesianMatrix<float> m(10000, 10000);
    for (auto _ : state) {
        m.batchRevisionY(2532, 6);
    }
}
BENCHMARK(batchRevisionYBenchmark);