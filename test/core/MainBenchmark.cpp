#include <benchmark/benchmark.h>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/CartesianMatrix.hpp"
#include "../../src/core/LatticeBoltzmannMethodD2Q9.h"
#include "../../src/core/LatticeBoltzmannMethodD2Q9.cpp"

static void LatticeBoltzmannMethodD2Q9_Initiation(benchmark::State& state) {
    CartesianMatrix<unsigned int> m1(4096, 4096, 1);
    for (auto _ : state) {
        LatticeBoltzmannMethodD2Q9 lbm (4096, 4096, 
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        m1.data, m1.data);
    }
}
BENCHMARK(LatticeBoltzmannMethodD2Q9_Initiation);

// static void LatticeBoltzmannMethodD2Q9_BuildResultMatrix(benchmark::State& state) {
//     CartesianMatrix<unsigned int> m1(4096, 4096, 1);
//     LatticeBoltzmannMethodD2Q9 lbm (4096, 4096, 
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
//         m1.data, m1.data);
//     for (auto _ : state) {
//         lbm.buildResultingDensityMatrix();
//         lbm.buildResultingTemperatureMatrix();
//     }
// }
// BENCHMARK(LatticeBoltzmannMethodD2Q9_BuildResultMatrix);

static void LatticeBoltzmannMethodD2Q9_Step(benchmark::State& state) {
    CartesianMatrix<unsigned int> m1(4096, 4096, 1);
    LatticeBoltzmannMethodD2Q9 lbm (4096, 4096, 
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        m1.data, m1.data);
    for (auto _ : state) {
        lbm.step();
    }
}
BENCHMARK(LatticeBoltzmannMethodD2Q9_Step);

// static void LatticeBoltzmannMethodD2Q9_Diffusion(benchmark::State& state) {
//     LatticeBoltzmannMethodD2Q9 lbm (8190, 8190, 
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
//         LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
//         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
//     for (auto _ : state) {
//         lbm.step();
//     }
// }
// BENCHMARK(LatticeBoltzmannMethodD2Q9_Diffusion);#include <benchmark/benchmark.h>

// static void OpenCLMain_COMPLEX(benchmark::State& state) {
//     CartesianMatrix<unsigned int> result(8192, 8192);
//     CartesianMatrix<unsigned int> m0(8192, 8192, 1);
//     CartesianMatrix<unsigned int> m1(8192, 8192, 2);
//     CartesianMatrix<unsigned int> m2(8192, 8192, 3);
//     CartesianMatrix<unsigned int> m3(8192, 8192, 4);
//     CartesianMatrix<unsigned int> m4(8192, 8192, 5);
//     CartesianMatrix<unsigned int> m5(8192, 8192, 6);
//     CartesianMatrix<unsigned int> m6(8192, 8192, 7);
//     CartesianMatrix<unsigned int> m7(8192, 8192, 8);
//     CartesianMatrix<unsigned int> m8(8192, 8192, 9);
//     OpenCLMain::instance();
//     for (auto _ : state) {
//         result.data = OpenCLMain::instance().evaluateArithmeticFormula(
//                 "A * 44 + B * 11 + C* 11 + D * 11 + E * 11 + F * 3 + G * 3 + H * 3 + I * 3",
//                 8192*8192,
//                 std::vector<unsigned int*>{m0.data.data(), 
//                                         m1.data.data(),
//                                         m2.data.data(),
//                                         m3.data.data(),
//                                         m4.data.data(),
//                                         m5.data.data(),
//                                         m6.data.data(),
//                                         m7.data.data(),
//                                         m8.data.data()});
//     }

// }
// BENCHMARK(OpenCLMain_COMPLEX);