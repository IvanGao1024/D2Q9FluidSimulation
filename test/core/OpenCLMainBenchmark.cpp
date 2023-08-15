#include <benchmark/benchmark.h>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/CartesianMatrix.hpp"

// static void OpenCLMain_Speed(benchmark::State& state) {
//     CartesianMatrix<unsigned int> m1(10000, 10000, 1);
//     CartesianMatrix<unsigned int> m2(10000, 10000, 2);
//     CartesianMatrix<unsigned int> m3(10000, 10000, 2);
//     OpenCLMain::instance();
//     for (auto _ : state) {
//         OpenCLMain::instance().arrayAddition(m1.data.data(), m2.data.data(), m3.data.data(), m1.mWidth*m1.mHeight);
//     }
// }
// BENCHMARK(OpenCLMain_Speed);

static void OpenCLMain_CMSpeed(benchmark::State& state) {
    CartesianMatrix<unsigned int> m1(10000, 10000, 1);
    CartesianMatrix<unsigned int> m2(10000, 10000, 2);
    CartesianMatrix<unsigned int> m3(10000, 10000, 2);
    for (auto _ : state) {
        m3 = m1 + m2;
    }
}
BENCHMARK(OpenCLMain_CMSpeed);
