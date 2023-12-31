#include <gtest/gtest.h>
#include "../../src/core/LatticeBoltzmannMethodD2Q9.h"
#include "../../src/core/LatticeBoltzmannMethodD2Q9.cpp"

class LatticeBoltzmannMethodD2Q9Test : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {

    }
};

TEST_F(LatticeBoltzmannMethodD2Q9Test, Diffusion) {
    Matrix<double> m1(8, 8, 0.25);
    LatticeBoltzmannMethodD2Q9 lbm (7, 7,
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
        m1.getShiftedData(), m1.getShiftedData());
    for (size_t i = 0; i < 10; i++)
    {
        lbm.step(true);
    }
}