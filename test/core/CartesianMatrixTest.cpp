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
    CartesianMatrix<int> m0;
    EXPECT_EQ(m0.getWidth(), 1);
    EXPECT_EQ(m0.getHeight(), 1);
    
    CartesianMatrix<int> m1(20,20);
    EXPECT_EQ(m1.at({5,5}), 0);
    CartesianMatrix<int> m2(20,20, 0);
    EXPECT_EQ(m2.at({5,5}), 0);
    CartesianMatrix<int> m3(20,20, 1);
    EXPECT_EQ(m3.at({5,5}), 1);
    m3.fill(0);
    EXPECT_EQ(m3.at({5,5}), 0);
    EXPECT_EQ(m3.at({19,19}), 0);

    CartesianMatrix<int> matrix0(3, 4, std::vector<std::vector<int>>{});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix0.at({j, i}), 0);
        }
    }
    CartesianMatrix<int> matrix1(3, 3, {{1,2,3},{3,2,1},{1,2,3}});
    CartesianMatrix<int> matrix2(3, 3, {{4,5,6},{6,5,4},{4,6,5}});
    CartesianMatrix<int> matrix3(3, 3, {{28, 33, 29}, {28, 31, 31}, {28, 33, 29}});
    CartesianMatrix<int> matrix4(3, 3, {{25, 30, 35}, {25, 30, 35}, {27, 30, 33}});
}

TEST_F(CartesianMatrixTest, MismatchedRowCount) {
    std::vector<std::vector<int>> values = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    EXPECT_THROW(CartesianMatrix<int>(4, 3, values), std::out_of_range);
}

TEST_F(CartesianMatrixTest, MismatchedColumnCount) {
    std::vector<std::vector<int>> values = {
        {1, 2, 3},
        {4, 5, 6, 7}, // This row has an extra column.
        {7, 8, 9}
    };

    EXPECT_THROW(CartesianMatrix<int>(3, 3, values), std::out_of_range);
}

// ==/at/[]/exception
TEST_F(CartesianMatrixTest, OperatorEqualsTest) {
    CartesianMatrix<int> m1(5,5);
    CartesianMatrix<int> m2(5,5);
    CartesianMatrix<int> m3(6,6);
    CartesianMatrix<int> m4(5,6);
    // ==
    EXPECT_TRUE(m1 == m2);
    EXPECT_EQ(m1, m2);
}

TEST_F(CartesianMatrixTest, OperatorExceptionTest){
    CartesianMatrix<int> m1(5,5);
    CartesianMatrix<int> m2(5,5);
    CartesianMatrix<int> m3(6,6);
    CartesianMatrix<int> m4(5,6);

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

TEST_F(CartesianMatrixTest, OperatorElementWiseTest){
    CartesianMatrix<int> m1(5,5);
    CartesianMatrix<int> m2(5,5);
    CartesianMatrix<int> m3(5,5);
    CartesianMatrix<int> m4(5,5);
    CartesianMatrix<int> m5(5,6);

    // * and +
    m1.batchRevisionX(1, 5);
    m1.batchRevisionY(1, 5);
    m1[{1, 1}] = 10;
    m2.batchRevisionX(1, 10);
    m3.batchRevisionX(1, 5);
    m4.batchRevisionY(1, 5);
    EXPECT_EQ(m1, m3 + m4);
    EXPECT_EQ(m2, m3 * 2);
    EXPECT_THROW(m3 + m5, std::out_of_range);
}

TEST_F(CartesianMatrixTest, OperatorMatrixMultiplicationTest){
    CartesianMatrix<int> matrix1(3, 3, {{1,2,3},{3,2,1},{1,2,3}});
    CartesianMatrix<int> matrix2(3, 3, {{4,5,6},{6,5,4},{4,6,5}});
    CartesianMatrix<int> matrix3(3, 3, {{28, 33, 29}, {28, 31, 31}, {28, 33, 29}});
    CartesianMatrix<int> matrix4(3, 3, {{25, 30, 35}, {25, 30, 35}, {27, 30, 33}});

    EXPECT_EQ(matrix1*matrix2, matrix3);
    EXPECT_EQ(matrix2*matrix1, matrix4);

    CartesianMatrix<int> matrix5(3, 4, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    CartesianMatrix<int> matrix6(4, 3, {{2, 3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}});
    CartesianMatrix<int> matrix7(3, 3, {{80, 90, 100}, {184, 210, 236}, {288, 330, 372}});
    CartesianMatrix<int> matrix8(4, 4, {{53, 62, 71, 80}, {98, 116, 134, 152}, {143, 170, 197, 224}, {188, 224, 260, 296}});
    CartesianMatrix<int> matrix9(4, 3, {{233, 270, 307}, {464, 540, 616}, {695, 810, 925}, {926, 1080, 1234}});
    

    EXPECT_EQ(matrix5*matrix6, matrix7);
    EXPECT_EQ(matrix6*matrix5, matrix8);
    EXPECT_EQ(matrix6*matrix4, matrix9);
    EXPECT_THROW(matrix4*matrix6, std::out_of_range);
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
    EXPECT_THROW(m1.getShift(static_cast<CartesianMatrix<int>::Direction>(999)), std::invalid_argument);
}


TEST_F(CartesianMatrixTest, FillxxTest) {
    CartesianMatrix<int> m1(5,5);
    EXPECT_EQ(m1.at({0,0}), 0);

    m1.fill(2);
    EXPECT_EQ(m1.at({0,0}), 2);
    EXPECT_EQ(m1.at({4,4}), 2);

    CartesianMatrix<int> m2(5,5);
    m1.fillRandom();
    m2.fillRandom();
    EXPECT_NE(m1, m2);

    CartesianMatrix<float> m3(5,5);
    m3.fillRandom();

    CartesianMatrix<double> m4(5,5);
    m4.fillRandom();
}