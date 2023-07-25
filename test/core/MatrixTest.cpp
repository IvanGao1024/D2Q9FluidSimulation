#include <gtest/gtest.h>
#include "../../src/core/Matrix.hpp"

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {

    }
};

TEST_F(MatrixTest, ConstructorInitiationTest) {
    Matrix<int> m1(20,20);
    EXPECT_EQ(m1.at({5,5}), 0);
    Matrix<int> m2(20,20, 0);
    EXPECT_EQ(m2.at({5,5}), 0);
    Matrix<int> m3(20,20, 1);
    EXPECT_EQ(m3.at({5,5}), 1);
}

TEST_F(MatrixTest, BaseShiftTest) {
    Matrix<int> m1(5,5);

    // int* n = q0_.Dequeue();
    // EXPECT_EQ(n, nullptr);

    // n = q1_.Dequeue();
    // ASSERT_NE(n, nullptr);
    // EXPECT_EQ(*n, 1);
    // EXPECT_EQ(q1_.size(), 0);
    // delete n;

    // n = q2_.Dequeue();
    // ASSERT_NE(n, nullptr);
    // EXPECT_EQ(*n, 2);
    // EXPECT_EQ(q2_.size(), 1);
    // delete n;
}