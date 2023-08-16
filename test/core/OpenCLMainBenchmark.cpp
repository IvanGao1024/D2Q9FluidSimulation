#include <benchmark/benchmark.h>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/CartesianMatrix.hpp"

static void OpenCLMain_Speed(benchmark::State& state) {
    CartesianMatrix<unsigned int> m1(10000, 10000, 1);
    CartesianMatrix<unsigned int> m2(10000, 10000, 2);
    std::vector<unsigned int>     result(100000000);
    OpenCLMain::instance();
    for (auto _ : state) {
        result = OpenCLMain::instance().evaluateArithmeticFormula("A + B + A + B + A + B", 100000000, std::vector<unsigned int*>{m1.data.data(), m2.data.data()});
    }
}
BENCHMARK(OpenCLMain_Speed);

static void OpenCLMain_CMSpeed(benchmark::State& state) {
    CartesianMatrix<unsigned int> m1(10000, 10000, 1);
    CartesianMatrix<unsigned int> m2(10000, 10000, 2);
    CartesianMatrix<unsigned int> result(10000, 10000);
    for (auto _ : state) {
        result = m1 + m2 + m1 + m2 + m1 + m2;
    }
}
BENCHMARK(OpenCLMain_CMSpeed);
