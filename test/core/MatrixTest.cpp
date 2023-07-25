#ifndef MATRIX_TEST
#define MATRIX_TEST

#include <gtest/gtest.h>

#include "../../src/core/Matrix.hpp"

class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {

    }
};

TEST_F(MatrixTest, IsEmptyInitially) {
//   EXPECT_EQ(q0_.size(), 0);
}

TEST_F(MatrixTest, DequeueWorks) {
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

// void MatrixTest::MatrixBasic ()
// {
// 	QBENCHMARK
// 	{
//    		Matrix<float> m(1000, 1000);
// 	}
   	
// 	Matrix<float> m(20, 20);
//     m[{5, 5}] = 42;
// 	qDebug() << m[{5, 5}];
//     // m[{5, 5}] = 42;
// 	// QVERIFY(m[{5, 5}] == 42);

//     // QString str = "Hello";
//     // QVERIFY(str.toUpper() == "HELLO");
// }

// void MatrixTest::MatrixShift ()
// {
//     // m.applyFunction(increment);
// 	// m.print();
//     // qDebug() << "-----";
//     // m.print();
//     // m.shift(Matrix<int>::Direction::E);  // shift to the right
//     // m.print();
//     // m.shift(Matrix<int>::Direction::S);  // shift down
//     // m.print();
//     // m.shift(Matrix<int>::Direction::SW); // shift down and to the left
//     // m.print();
// }

#endif //MATRIX_TEST
