#include <gtest/gtest.h>
#include "../../src/core/Matrix.hpp"

class MatrixTest : public ::testing::Test {
public:
    Matrix<int> matrix0;
    Matrix<int> matrix1;
    Matrix<int> matrix2;
    Matrix<int> matrix3;
    Matrix<int> matrix4;
    Matrix<int> matrix5;
    Matrix<int> matrix6;
    Matrix<int> matrix7;
    Matrix<int> matrix8;

protected:
    void SetUp() override {
        matrix0 = Matrix<int>(3, 3);

        matrix1 = Matrix<int>(3, 3);
        matrix1.shift(1, 0);
        EXPECT_EQ(matrix1.getRowShiftIndex(), 0);
        EXPECT_EQ(matrix1.getColShiftIndex(), 1);

        matrix2 = Matrix<int>(3, 3);
        matrix2.shift(0, 1);
        EXPECT_EQ(matrix2.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix2.getColShiftIndex(), 0);

        matrix3 = Matrix<int>(3, 3);
        matrix3.shift(-1, 0);
        EXPECT_EQ(matrix3.getRowShiftIndex(), 0);
        EXPECT_EQ(matrix3.getColShiftIndex(), 2);

        matrix4 = Matrix<int>(3, 3);
        matrix4.shift(0, -1);
        EXPECT_EQ(matrix4.getRowShiftIndex(), 2);
        EXPECT_EQ(matrix4.getColShiftIndex(), 0);

        matrix5 = Matrix<int>(3, 3);
        matrix5.shift(1, 1);
        EXPECT_EQ(matrix5.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix5.getColShiftIndex(), 1);

        matrix6 = Matrix<int>(3, 3);
        matrix6.shift(-1, 1);
        EXPECT_EQ(matrix6.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix6.getColShiftIndex(), 2);

        matrix7 = Matrix<int>(3, 3);
        matrix7.shift(-1, -1);
        EXPECT_EQ(matrix7.getRowShiftIndex(), 2);
        EXPECT_EQ(matrix7.getColShiftIndex(), 2);

        matrix8 = Matrix<int>(3, 3);
        matrix8.shift(1, -1);
        EXPECT_EQ(matrix8.getRowShiftIndex(), 2);
        EXPECT_EQ(matrix8.getColShiftIndex(), 1);
    }

    void TearDown() override {

    }
};

TEST_F(MatrixTest, ConstructorInitiationTest) {
    Matrix<int> m0;
    EXPECT_EQ(m0.getRowShiftIndex(), 0);
    EXPECT_EQ(m0.getColShiftIndex(), 0);
    auto expected = std::make_pair(0u, 0u); // 'u' for unsigned int
    auto result = m0.getShiftIndexPair();
    EXPECT_EQ(expected, result);
    EXPECT_EQ(m0.getM(), 1);
    EXPECT_EQ(m0.getN(), 1);
    EXPECT_EQ(m0.getLength(), 1);
    EXPECT_EQ(m0.getShiftedData().at(0), 0);
    
    Matrix<int> m1(20,20);
    EXPECT_EQ(m1.getRowShiftIndex(), 0);
    EXPECT_EQ(m1.getColShiftIndex(), 0);
    EXPECT_EQ(m1.getM(), 20);
    EXPECT_EQ(m1.getN(), 20);
    EXPECT_EQ(m1.getLength(), 400);
#pragma omp parallel for
    for (int i = 0; i < m1.getLength(); ++i) {
        EXPECT_EQ(m1.getShiftedData().at(i), 0);
    }

    Matrix<int> m2(20,20, 0);
    EXPECT_EQ(m2.getRowShiftIndex(), 0);
    EXPECT_EQ(m2.getColShiftIndex(), 0);
    EXPECT_EQ(m2.getM(), 20);
    EXPECT_EQ(m2.getN(), 20);
    EXPECT_EQ(m2.getLength(), 400);
#pragma omp parallel for
    for (int i = 0; i < m2.getLength(); ++i) {
        EXPECT_EQ(m2.getShiftedData().at(i), 0);
    }

    Matrix<int> m3(20,20, 1);
    EXPECT_EQ(m3.getRowShiftIndex(), 0);
    EXPECT_EQ(m3.getColShiftIndex(), 0);
    EXPECT_EQ(m3.getM(), 20);
    EXPECT_EQ(m3.getN(), 20);
    EXPECT_EQ(m3.getLength(), 400);
#pragma omp parallel for
    for (int i = 0; i < m3.getLength(); ++i) {
        EXPECT_EQ(m3.getShiftedData().at(i), 1);
    }

    EXPECT_THROW(Matrix<int> m4(3, 4, std::vector<int>{}), std::invalid_argument);    

    Matrix<int> matrix1(3, 3, {1,2,3,3,2,1,1,2,3});
    // matrix1.print();
    Matrix<int> matrix2(3, 3, {4,5,6,6,5,4,4,6,5});
    // matrix2.print();
    Matrix<int> matrix3(3, 3, {28, 33, 29, 28, 31, 31, 28, 33, 29});
    // matrix3.print();
    Matrix<int> matrix4(3, 3, {25, 30, 35, 25, 30, 35, 27, 30, 33});
    // matrix4.print();
}

TEST_F(MatrixTest, Shift) {
    Matrix<int> m0(3, 3, {0, 0, 0, 0, 1, 0, 0, 0, 0});
    Matrix<int> m1(3, 3, {0, 0, 0, 0, 0, 1, 0, 0, 0});
    Matrix<int> m2(3, 3, {0, 1, 0, 0, 0, 0, 0, 0, 0});
    Matrix<int> m3(3, 3, {0, 0, 0, 1, 0, 0, 0, 0, 0});
    Matrix<int> m4(3, 3, {0, 0, 0, 0, 0, 0, 0, 1, 0});
    Matrix<int> m5(3, 3, {0, 0, 1, 0, 0, 0, 0, 0, 0});
    Matrix<int> m6(3, 3, {1, 0, 0, 0, 0, 0, 0, 0, 0});
    Matrix<int> m7(3, 3, {0, 0, 0, 0, 0, 0, 1, 0, 0});
    Matrix<int> m8(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 1});
    EXPECT_EQ(m0.getShiftedData(1, 0), m1.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(0, 1), m2.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(-1, 0), m3.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(0, -1), m4.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(1, 1), m5.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(-1, 1), m6.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(-1, -1), m7.getShiftedData());
    EXPECT_EQ(m0.getShiftedData(1, -1), m8.getShiftedData());

    EXPECT_EQ(m1.getShiftedData(-1, 0), m0.getShiftedData());
    EXPECT_EQ(m2.getShiftedData(0, -1), m0.getShiftedData());
    EXPECT_EQ(m3.getShiftedData(1, 0), m0.getShiftedData());
    EXPECT_EQ(m4.getShiftedData(0, 1), m0.getShiftedData());
    EXPECT_EQ(m5.getShiftedData(-1, -1), m0.getShiftedData());
    EXPECT_EQ(m6.getShiftedData(1, -1), m0.getShiftedData());
    EXPECT_EQ(m7.getShiftedData(1, 1), m0.getShiftedData());
    EXPECT_EQ(m8.getShiftedData(-1, 1), m0.getShiftedData());
}

TEST_F(MatrixTest, Fill) {
    Matrix<int> m0(3, 3, {25, 30, 35, 25, 30, 35, 27, 30, 33});
    Matrix<int> m1(3, 3);
    m0.fill(0);
    EXPECT_EQ(m0.getShiftedData(), m1.getShiftedData());
}

TEST_F(MatrixTest, IndexRevision) {
    Matrix<int> m0(3, 3, {1, 0, 0, 0, 0, 0, 0, 0, 0});
    Matrix<int> m1(3, 3);
    m1.indexRevision(0, 0, 1);
    EXPECT_EQ(m1.getShiftedData(), m0.getShiftedData());

    matrix1.indexRevision(0, 0, 1);
    EXPECT_EQ(matrix1.getShiftedData(), m0.getShiftedData());
}

TEST_F(MatrixTest, RowRevision) {
    Matrix<int> m0(3, 3, {1, 1, 1, 0, 0, 0, 0, 0, 0});
    Matrix<int> mBasic(3, 3);
    mBasic.rowRevision(0, 1);
    EXPECT_EQ(mBasic.getShiftedData(), m0.getShiftedData());

    Matrix<int> m1(3, 3);
    m1.shift(1, 0);
    m1.rowRevision(0, 1);
    EXPECT_EQ(m1.getShiftedData(), m0.getShiftedData());

    Matrix<int> m2(3, 3);
    m2.shift(0, 1);
    m2.rowRevision(0, 1);
    EXPECT_EQ(m2.getShiftedData(), m0.getShiftedData());

    Matrix<int> m3(3, 3);
    m3.shift(-1, 0);
    m3.rowRevision(0, 1);
    EXPECT_EQ(m2.getShiftedData(), m0.getShiftedData());

    Matrix<int> m4(3, 3);
    m4.shift(0, -1);
    m4.rowRevision(0, 1);
    EXPECT_EQ(m4.getShiftedData(), m0.getShiftedData());

    Matrix<int> m5(3, 3);
    m5.shift(1, 1);
    m5.rowRevision(0, 1);
    EXPECT_EQ(m5.getShiftedData(), m0.getShiftedData());

    Matrix<int> m6(3, 3);
    m6.shift(-1, 1);
    m6.rowRevision(0, 1);
    EXPECT_EQ(m6.getShiftedData(), m0.getShiftedData());

    Matrix<int> m7(3, 3);
    m7.shift(-1, -1);
    m7.rowRevision(0, 1);
    EXPECT_EQ(m7.getShiftedData(), m0.getShiftedData());

    Matrix<int> m8(3, 3);
    m8.shift(1, -1);
    m8.rowRevision(0, 1);
    EXPECT_EQ(m8.getShiftedData(), m0.getShiftedData());
}

TEST_F(MatrixTest, ColRevision) {
    Matrix<int> m0(3, 3, {1, 0, 0, 1, 0, 0, 1, 0, 0});
    Matrix<int> mBasic(3, 3);
    mBasic.colRevision(0, 1);
    EXPECT_EQ(mBasic.getShiftedData(), m0.getShiftedData());

    Matrix<int> m1(3, 3);
    m1.shift(1, 0);
    m1.colRevision(0, 1);
    EXPECT_EQ(m1.getShiftedData(), m0.getShiftedData());

    Matrix<int> m2(3, 3);
    m2.shift(0, 1);
    m2.colRevision(0, 1);
    EXPECT_EQ(m2.getShiftedData(), m0.getShiftedData());

    Matrix<int> m3(3, 3);
    m3.shift(-1, 0);
    m3.colRevision(0, 1);
    EXPECT_EQ(m2.getShiftedData(), m0.getShiftedData());

    Matrix<int> m4(3, 3);
    m4.shift(0, -1);
    m4.colRevision(0, 1);
    EXPECT_EQ(m4.getShiftedData(), m0.getShiftedData());

    Matrix<int> m5(3, 3);
    m5.shift(1, 1);
    m5.colRevision(0, 1);
    EXPECT_EQ(m5.getShiftedData(), m0.getShiftedData());

    Matrix<int> m6(3, 3);
    m6.shift(-1, 1);
    m6.colRevision(0, 1);
    EXPECT_EQ(m6.getShiftedData(), m0.getShiftedData());

    Matrix<int> m7(3, 3);
    m7.shift(-1, -1);
    m7.colRevision(0, 1);
    EXPECT_EQ(m7.getShiftedData(), m0.getShiftedData());

    Matrix<int> m8(3, 3);
    m8.shift(1, -1);
    m8.colRevision(0, 1);
    EXPECT_EQ(m8.getShiftedData(), m0.getShiftedData());
}

// TEST_F(CartesianMatrixTest, MismatchedRowCount) {
//     std::vector<std::vector<int>> values = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 9}
//     };

//     EXPECT_THROW(Matrix<int>(4, 3, values), std::out_of_range);
// }

// TEST_F(CartesianMatrixTest, MismatchedColumnCount) {
//     std::vector<std::vector<int>> values = {
//         {1, 2, 3},
//         {4, 5, 6, 7}, // This row has an extra column.
//         {7, 8, 9}
//     };

//     EXPECT_THROW(Matrix<int>(3, 3, values), std::out_of_range);
// }

// // ==/at/[]/exception
// TEST_F(CartesianMatrixTest, OperatorEqualsTest) {
//     Matrix<int> m1(5,5);
//     Matrix<int> m2(5,5);
//     Matrix<int> m3(6,6);
//     Matrix<int> m4(5,6);
//     // ==
//     EXPECT_TRUE(m1 == m2);
//     EXPECT_EQ(m1, m2);
// }

// TEST_F(CartesianMatrixTest, OperatorExceptionTest){
//     Matrix<int> m1(5,5);
//     Matrix<int> m2(5,5);
//     Matrix<int> m3(6,6);
//     Matrix<int> m4(5,6);

//     // exception
//     EXPECT_THROW(m1.at({5,5}), std::out_of_range);
//     EXPECT_THROW(m1.at({4,5}), std::out_of_range);
//     EXPECT_THROW(m1.at({5,4}), std::out_of_range);
//     EXPECT_THROW(m1.at({-1,3}), std::out_of_range);
//     EXPECT_THROW(m1.at({3,-1}), std::out_of_range);
//     EXPECT_THROW(m1.batchRevisionX(5, 1), std::out_of_range);
//     EXPECT_NO_THROW(m1.batchRevisionX(4, 1));
//     EXPECT_THROW(m1.batchRevisionY(5, 1), std::out_of_range);
//     EXPECT_NO_THROW(m1.batchRevisionY(4, 1));
//     EXPECT_NO_THROW(m1.at({4,4}));

//     // matrix ordering test and []
//     m1[{4,4}] = 5;
//     // m1.print();
//     EXPECT_EQ(m1.at({4,4}), 5);
//     EXPECT_NE(m1, m2);
//     EXPECT_NE(m1, m3);
//     EXPECT_NE(m1, m4);
// }

// TEST_F(CartesianMatrixTest, OperatorElementWiseTest){
//     Matrix<int> m1(5,5);
//     Matrix<int> m2(5,5);
//     Matrix<int> m3(5,5);
//     Matrix<int> m4(5,5);
//     Matrix<int> m5(5,6);

//     // * and +
//     m1.batchRevisionX(1, 5);
//     m1.batchRevisionY(1, 5);
//     m1[{1, 1}] = 10;
//     m2.batchRevisionX(1, 10);
//     m3.batchRevisionX(1, 5);
//     m4.batchRevisionY(1, 5);
//     EXPECT_EQ(m1, m3 + m4);
//     EXPECT_EQ(m2, m3 * 2);
//     EXPECT_EQ(m3, m1 - m4);
//     EXPECT_THROW(m3 - m5, std::out_of_range);
//     EXPECT_EQ(m1 + 1, m3 + m4 + 1);
//     EXPECT_THROW(m3 + m5, std::out_of_range);
// }

// TEST_F(CartesianMatrixTest, OperatorMatrixMultiplicationTest){
//     Matrix<int> matrix1(3, 3, {{1,2,3},{3,2,1},{1,2,3}});
//     Matrix<int> matrix2(3, 3, {{4,5,6},{6,5,4},{4,6,5}});
//     Matrix<int> matrix3(3, 3, {{28, 33, 29}, {28, 31, 31}, {28, 33, 29}});
//     Matrix<int> matrix4(3, 3, {{25, 30, 35}, {25, 30, 35}, {27, 30, 33}});

//     EXPECT_EQ(matrix1*matrix2, matrix3);
//     EXPECT_EQ(matrix2*matrix1, matrix4);

//     Matrix<int> matrix5(3, 4, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
//     Matrix<int> matrix6(4, 3, {{2, 3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}});
//     Matrix<int> matrix7(3, 3, {{80, 90, 100}, {184, 210, 236}, {288, 330, 372}});
//     Matrix<int> matrix8(4, 4, {{53, 62, 71, 80}, {98, 116, 134, 152}, {143, 170, 197, 224}, {188, 224, 260, 296}});
//     Matrix<int> matrix9(4, 3, {{233, 270, 307}, {464, 540, 616}, {695, 810, 925}, {926, 1080, 1234}});
    

//     EXPECT_EQ(matrix5*matrix6, matrix7);
//     EXPECT_EQ(matrix6*matrix5, matrix8);
//     EXPECT_EQ(matrix6*matrix4, matrix9);
//     EXPECT_THROW(matrix4*matrix6, std::out_of_range);
// }

// TEST_F(CartesianMatrixTest, BatchRevisionTest){
//     Matrix<int> m1(5,5);
//     m1.batchRevisionY(1, 5);
//     // m1.print();
//     m1.batchRevisionX(1, 6);
//     // m1.print();
//     EXPECT_EQ(m1.at({1,1}), 6);
//     EXPECT_EQ(m1.at({0,1}), 5);
//     EXPECT_EQ(m1.at({4,1}), 5);
//     EXPECT_EQ(m1.at({1,0}), 6);
//     EXPECT_EQ(m1.at({1,4}), 6);
//     // m1.print();
// }

// TEST_F(CartesianMatrixTest, BaseShiftTest) {
//     Matrix<int> m1(5,5);
//     m1.batchRevisionY(0, 1);
//     EXPECT_EQ(m1.at({0,0}), 1);
//     m1.baseShift(Matrix<int>::Direction::UP);
//     EXPECT_EQ(m1.at({0,0}), 0);
//     EXPECT_EQ(m1.at({0,1}), 1);
//     m1.baseShift(Matrix<int>::Direction::UP);
//     EXPECT_EQ(m1.at({0,1}), 0);
//     EXPECT_EQ(m1.at({0,2}), 1);
//     m1.baseShift(Matrix<int>::Direction::UP);
//     EXPECT_EQ(m1.at({0,2}), 0);
//     EXPECT_EQ(m1.at({0,3}), 1);
//     m1.baseShift(Matrix<int>::Direction::UP);
//     EXPECT_EQ(m1.at({0,3}), 0);
//     EXPECT_EQ(m1.at({0,4}), 1);
//     m1.baseShift(Matrix<int>::Direction::UP);
//     EXPECT_EQ(m1.at({0,4}), 0);
//     EXPECT_EQ(m1.at({0,0}), 1);

//     Matrix<int> m2(5,5);
//     m2.batchRevisionY(4, 1);
//     EXPECT_EQ(m2.at({0,4}), 1);
//     m2.baseShift(Matrix<int>::Direction::DOWN);
//     EXPECT_EQ(m2.at({0,4}), 0);
//     EXPECT_EQ(m2.at({0,3}), 1);
//     m2.baseShift(Matrix<int>::Direction::DOWN);
//     EXPECT_EQ(m2.at({0,3}), 0);
//     EXPECT_EQ(m2.at({0,2}), 1);
//     m2.baseShift(Matrix<int>::Direction::DOWN);
//     EXPECT_EQ(m2.at({0,2}), 0);
//     EXPECT_EQ(m2.at({0,1}), 1);
//     m2.baseShift(Matrix<int>::Direction::DOWN);
//     EXPECT_EQ(m2.at({0,1}), 0);
//     EXPECT_EQ(m2.at({0,0}), 1);
//     m2.baseShift(Matrix<int>::Direction::DOWN);
//     EXPECT_EQ(m2.at({0,0}), 0);
//     EXPECT_EQ(m2.at({0,4}), 1);

//     Matrix<int> m3(5,5);
//     m3.batchRevisionX(4, 1);
//     EXPECT_EQ(m3.at({4,0}), 1);
//     m3.baseShift(Matrix<int>::Direction::LEFT);
//     EXPECT_EQ(m3.at({4,0}), 0);
//     EXPECT_EQ(m3.at({3,0}), 1);
//     m3.baseShift(Matrix<int>::Direction::LEFT);
//     EXPECT_EQ(m3.at({3,0}), 0);
//     EXPECT_EQ(m3.at({2,0}), 1);
//     m3.baseShift(Matrix<int>::Direction::LEFT);
//     EXPECT_EQ(m3.at({2,0}), 0);
//     EXPECT_EQ(m3.at({1,0}), 1);
//     m3.baseShift(Matrix<int>::Direction::LEFT);
//     EXPECT_EQ(m3.at({1,0}), 0);
//     EXPECT_EQ(m3.at({0,0}), 1);
//     m3.baseShift(Matrix<int>::Direction::LEFT);
//     EXPECT_EQ(m3.at({0,0}), 0);
//     EXPECT_EQ(m3.at({4,0}), 1);

//     Matrix<int> m4(5,5);
//     m4.batchRevisionX(0, 1);
//     EXPECT_EQ(m4.at({0,0}), 1);
//     m4.baseShift(Matrix<int>::Direction::RIGHT);
//     EXPECT_EQ(m4.at({0,0}), 0);
//     EXPECT_EQ(m4.at({1,0}), 1);
//     m4.baseShift(Matrix<int>::Direction::RIGHT);
//     EXPECT_EQ(m4.at({1,0}), 0);
//     EXPECT_EQ(m4.at({2,0}), 1);
//     m4.baseShift(Matrix<int>::Direction::RIGHT);
//     EXPECT_EQ(m4.at({2,0}), 0);
//     EXPECT_EQ(m4.at({3,0}), 1);
//     m4.baseShift(Matrix<int>::Direction::RIGHT);
//     EXPECT_EQ(m4.at({3,0}), 0);
//     EXPECT_EQ(m4.at({4,0}), 1);
//     m4.baseShift(Matrix<int>::Direction::RIGHT);
//     EXPECT_EQ(m4.at({4,0}), 0);
//     EXPECT_EQ(m4.at({0,0}), 1);

//     // Cast an out-of-range integer to Direction to force test on the default case:
//     EXPECT_THROW(m1.getShift(static_cast<Matrix<int>::Direction>(999)), std::invalid_argument);
// }


// TEST_F(CartesianMatrixTest, FillxxTest) {
//     Matrix<int> m1(5,5);
//     EXPECT_EQ(m1.at({0,0}), 0);

//     m1.fill(2);
//     EXPECT_EQ(m1.at({0,0}), 2);
//     EXPECT_EQ(m1.at({4,4}), 2);

//     Matrix<int> m2(5,5);
//     m1.fillRandom();
//     m2.fillRandom();
//     EXPECT_NE(m1, m2);

//     MatrixTest<float> m3(5,5);
//     m3.fillRandom();

//     MatrixTest<double> m4(5,5);
//     m4.fillRandom();
// }