#include <benchmark/benchmark.h>
#include "../../src/core/LatticeBoltzmannMethodD2Q9.h"
#include "../../src/core/LatticeBoltzmannMethodD2Q9.cpp"

static void InitiationBenchmark(benchmark::State& state) {
    for (auto _ : state) {
    LatticeBoltzmannMethodD2Q9 lbm (10000, 10000, nullptr, nullptr, nullptr, 0.15, 0.15);
    }
}
BENCHMARK(InitiationBenchmark);