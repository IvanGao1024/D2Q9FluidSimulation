#include <gtest/gtest.h>
#include "../../src/core/CartesianMatrix.hpp"

class CartesianMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {

    }
};

// constructor/at
TEST_F(CartesianMatrixTest, ConstructorInitiationTest) {
    CartesianMatrix<int> m1(20,20);
    EXPECT_EQ(m1.at({5,5}), 0);
    CartesianMatrix<int> m2(20,20, 0);
    EXPECT_EQ(m2.at({5,5}), 0);
    CartesianMatrix<int> m3(20,20, 1);
    EXPECT_EQ(m3.at({5,5}), 1);
}

// ==/at/[]/exception
TEST_F(CartesianMatrixTest, OperatorTest) {
    CartesianMatrix<int> m1(5,5);
    CartesianMatrix<int> m2(5,5);
    CartesianMatrix<int> m3(6,6);
    CartesianMatrix<int> m4(5,6);

    // ==
    EXPECT_TRUE(m1 == m2);
    EXPECT_EQ(m1, m2);

    // exception
    EXPECT_THROW(m1.at({5,5}), std::out_of_range);
    EXPECT_THROW(m1.at({4,5}), std::out_of_range);
    EXPECT_THROW(m1.at({5,4}), std::out_of_range);
    EXPECT_THROW(m1.at({-1,3}), std::out_of_range);
    EXPECT_THROW(m1.at({3,-1}), std::out_of_range);
    EXPECT_THROW(m1.batchRevisionX(5, 1), std::out_of_range);
    EXPECT_NO_THROW(m1.batchRevisionX(4, 1));
    EXPECT_THROW(m1.batchRevisionY(5, 1), std::out_of_range);
    EXPECT_NO_THROW(m1.batchRevisionY(4, 1));
    EXPECT_NO_THROW(m1.at({4,4}));

    // matrix ordering test and []
    m1[{4,4}] = 5;
    // m1.print();
    EXPECT_EQ(m1.at({4,4}), 5);
    EXPECT_NE(m1, m2);
    EXPECT_NE(m1, m3);
    EXPECT_NE(m1, m4);
}

TEST_F(CartesianMatrixTest, BatchRevisionTest){
    CartesianMatrix<int> m1(5,5);
    m1.batchRevisionY(1, 5);
    // m1.print();
    m1.batchRevisionX(1, 6);
    // m1.print();
    EXPECT_EQ(m1.at({1,1}), 6);
    EXPECT_EQ(m1.at({0,1}), 5);
    EXPECT_EQ(m1.at({4,1}), 5);
    EXPECT_EQ(m1.at({1,0}), 6);
    EXPECT_EQ(m1.at({1,4}), 6);
    // m1.print();
}

TEST_F(CartesianMatrixTest, BaseShiftTest) {
    CartesianMatrix<int> m1(5,5);
    m1.batchRevisionY(0, 1);
    EXPECT_EQ(m1.at({0,0}), 1);
    m1.baseShift(CartesianMatrix<int>::Direction::UP);
    EXPECT_EQ(m1.at({0,0}), 0);
    EXPECT_EQ(m1.at({0,1}), 1);
    m1.baseShift(CartesianMatrix<int>::Direction::UP);
    EXPECT_EQ(m1.at({0,1}), 0);
    EXPECT_EQ(m1.at({0,2}), 1);
    m1.baseShift(CartesianMatrix<int>::Direction::UP);
    EXPECT_EQ(m1.at({0,2}), 0);
    EXPECT_EQ(m1.at({0,3}), 1);
    m1.baseShift(CartesianMatrix<int>::Direction::UP);
    EXPECT_EQ(m1.at({0,3}), 0);
    EXPECT_EQ(m1.at({0,4}), 1);
    m1.baseShift(CartesianMatrix<int>::Direction::UP);
    EXPECT_EQ(m1.at({0,4}), 0);
    EXPECT_EQ(m1.at({0,0}), 1);

    CartesianMatrix<int> m2(5,5);
    m2.batchRevisionY(4, 1);
    EXPECT_EQ(m2.at({0,4}), 1);
    m2.baseShift(CartesianMatrix<int>::Direction::DOWN);
    EXPECT_EQ(m2.at({0,4}), 0);
    EXPECT_EQ(m2.at({0,3}), 1);
    m2.baseShift(CartesianMatrix<int>::Direction::DOWN);
    EXPECT_EQ(m2.at({0,3}), 0);
    EXPECT_EQ(m2.at({0,2}), 1);
    m2.baseShift(CartesianMatrix<int>::Direction::DOWN);
    EXPECT_EQ(m2.at({0,2}), 0);
    EXPECT_EQ(m2.at({0,1}), 1);
    m2.baseShift(CartesianMatrix<int>::Direction::DOWN);
    EXPECT_EQ(m2.at({0,1}), 0);
    EXPECT_EQ(m2.at({0,0}), 1);
    m2.baseShift(CartesianMatrix<int>::Direction::DOWN);
    EXPECT_EQ(m2.at({0,0}), 0);
    EXPECT_EQ(m2.at({0,4}), 1);

    CartesianMatrix<int> m3(5,5);
    m3.batchRevisionX(4, 1);
    EXPECT_EQ(m3.at({4,0}), 1);
    m3.baseShift(CartesianMatrix<int>::Direction::LEFT);
    EXPECT_EQ(m3.at({4,0}), 0);
    EXPECT_EQ(m3.at({3,0}), 1);
    m3.baseShift(CartesianMatrix<int>::Direction::LEFT);
    EXPECT_EQ(m3.at({3,0}), 0);
    EXPECT_EQ(m3.at({2,0}), 1);
    m3.baseShift(CartesianMatrix<int>::Direction::LEFT);
    EXPECT_EQ(m3.at({2,0}), 0);
    EXPECT_EQ(m3.at({1,0}), 1);
    m3.baseShift(CartesianMatrix<int>::Direction::LEFT);
    EXPECT_EQ(m3.at({1,0}), 0);
    EXPECT_EQ(m3.at({0,0}), 1);
    m3.baseShift(CartesianMatrix<int>::Direction::LEFT);
    EXPECT_EQ(m3.at({0,0}), 0);
    EXPECT_EQ(m3.at({4,0}), 1);

    CartesianMatrix<int> m4(5,5);
    m4.batchRevisionX(0, 1);
    EXPECT_EQ(m4.at({0,0}), 1);
    m4.baseShift(CartesianMatrix<int>::Direction::RIGHT);
    EXPECT_EQ(m4.at({0,0}), 0);
    EXPECT_EQ(m4.at({1,0}), 1);
    m4.baseShift(CartesianMatrix<int>::Direction::RIGHT);
    EXPECT_EQ(m4.at({1,0}), 0);
    EXPECT_EQ(m4.at({2,0}), 1);
    m4.baseShift(CartesianMatrix<int>::Direction::RIGHT);
    EXPECT_EQ(m4.at({2,0}), 0);
    EXPECT_EQ(m4.at({3,0}), 1);
    m4.baseShift(CartesianMatrix<int>::Direction::RIGHT);
    EXPECT_EQ(m4.at({3,0}), 0);
    EXPECT_EQ(m4.at({4,0}), 1);
    m4.baseShift(CartesianMatrix<int>::Direction::RIGHT);
    EXPECT_EQ(m4.at({4,0}), 0);
    EXPECT_EQ(m4.at({0,0}), 1);

    // Cast an out-of-range integer to Direction to force test on the default case:
    auto shiftDefault = m1.getShift(static_cast<CartesianMatrix<int>::Direction>(999));
    EXPECT_EQ(shiftDefault, std::make_pair(0, 0));

}