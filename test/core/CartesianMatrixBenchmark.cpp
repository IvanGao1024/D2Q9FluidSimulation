#include <benchmark/benchmark.h>
#include "../../src/core/CartesianMatrix.hpp"
class CartesianMatrixFixture : public ::benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        m = new CartesianMatrix<double>(10000, 10000);
    }

    void TearDown(const ::benchmark::State& state) {
        delete m;
    }

protected:
    CartesianMatrix<double>* m;
};

BENCHMARK_DEFINE_F(CartesianMatrixFixture, LargeSizeInitialization)
(benchmark::State& state) {
    for (auto _ : state) {
        *m = CartesianMatrix<double>(10000, 10000, 1.0);
    }
}
BENCHMARK_REGISTER_F(CartesianMatrixFixture, LargeSizeInitialization);

BENCHMARK_DEFINE_F(CartesianMatrixFixture, ElementWiseMultiplication)
(benchmark::State& state) {
    for (auto _ : state) {
        *m * 0.333;
    }
}
BENCHMARK_REGISTER_F(CartesianMatrixFixture, ElementWiseMultiplication);

BENCHMARK_DEFINE_F(CartesianMatrixFixture, BatchRevisionBenchmark)
(benchmark::State& state) {
    for (auto _ : state) {
        m->batchRevisionX(2532, 6);
        m->batchRevisionY(2532, 6);
    }
}
BENCHMARK_REGISTER_F(CartesianMatrixFixture, BatchRevisionBenchmark);