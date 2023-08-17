#include <benchmark/benchmark.h>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/CartesianMatrix.hpp"

static void OpenCLMain(benchmark::State& state) {
    CartesianMatrix<unsigned int> m1(10000, 10000, 1);
    CartesianMatrix<unsigned int> m2(10000, 10000, 2);
    CartesianMatrix<unsigned int> m3(10000, 10000, 3);
    CartesianMatrix<unsigned int> m4(10000, 10000, 4);
    CartesianMatrix<unsigned int> m5(10000, 10000, 5);
    OpenCLMain::instance();
    std::vector<unsigned int>     result(10000*10000);
    for (auto _ : state) {
        OpenCLMain::instance().evaluateArithmeticFormula(
            "E + 3 + A * (B - 4 / 2) + (C / 3) * (7 - E) + (E + 3) / D - 9",
            10000*10000,
            std::vector<unsigned int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data(), m5.data.data()});
    }
}
BENCHMARK(OpenCLMain);
