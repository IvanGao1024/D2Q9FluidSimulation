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
        // LatticeBoltzmannMethodD2Q9 lbm (24, 24, 
        // LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        // LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
        // LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
        // LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
        // nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        // lbm.step();
        // lbm.mResultingDensityMatrix.print();
        // lbm.mResultingTemperatureMatrix.print();
}
