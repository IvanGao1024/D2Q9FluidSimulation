#include <benchmark/benchmark.h>
#include "../../src/core/LatticeBoltzmannMethodD2Q9.h"
#include "../../src/core/LatticeBoltzmannMethodD2Q9.cpp"

static void LatticeBoltzmannMethodD2Q9_Initiation(benchmark::State& state) {
    for (auto _ : state) {
        LatticeBoltzmannMethodD2Q9 lbm (5000, 5000, 
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
}
BENCHMARK(LatticeBoltzmannMethodD2Q9_Initiation);

static void LatticeBoltzmannMethodD2Q9_Diffusion(benchmark::State& state) {
    for (auto _ : state) {
        LatticeBoltzmannMethodD2Q9 lbm (100, 100, 
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
}
BENCHMARK(LatticeBoltzmannMethodD2Q9_Diffusion);