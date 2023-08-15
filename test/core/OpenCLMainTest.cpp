#include <gtest/gtest.h>
#include <deque>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/CartesianMatrix.hpp"
class OpenCLMainTest : public ::testing::Test {
};

std::string queueToString(const std::queue<std::string>& q) {
    std::queue<std::string> copy = q;
    std::string result;
    while (!copy.empty()) {
        result += copy.front() + " ";
        copy.pop();
    }
    return result;
}

TEST_F(OpenCLMainTest, ShuntingYardTest_SimpleAddition) {
    std::queue<std::string> result = OpenCLMain::instance().enqueueArithmeticFormula("A+B");
    EXPECT_EQ(queueToString(result), "A B + ");
}

TEST_F(OpenCLMainTest, ShuntingYardTest_ComplexExpression) {
    std::queue<std::string> result = OpenCLMain::instance().enqueueArithmeticFormula("1/A*(B-3)/C-(20*D+E+1)");
    EXPECT_EQ(queueToString(result), "1 A / B 3 - * C / 20 D * E + 1 + - ");
}

TEST_F(OpenCLMainTest, ShuntingYardTest_SimpleDivision) {
    std::queue<std::string> result = OpenCLMain::instance().enqueueArithmeticFormula("A/2");
    EXPECT_EQ(queueToString(result), "A 2 / ");
}

TEST_F(OpenCLMainTest, ShuntingYardTest_MultipleVariables) {
    std::queue<std::string> result = OpenCLMain::instance().enqueueArithmeticFormula("A*2+B/34+C");
    EXPECT_EQ(queueToString(result), "A 2 * B 34 / + C + ");
}

TEST_F(OpenCLMainTest, ShuntingYardTest_ExpressionWithParentheses) {
    std::queue<std::string> result = OpenCLMain::instance().enqueueArithmeticFormula("(A+B)*C");
    EXPECT_EQ(queueToString(result), "A B + C * ");
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest) {
    CartesianMatrix<unsigned int> m1(10000, 10000, 1);
    CartesianMatrix<unsigned int> m2(10000, 10000, 2);
    CartesianMatrix<unsigned int> m3(10000, 10000, 2);
    CartesianMatrix<unsigned int> m4(10000, 10000, 2);
    CartesianMatrix<unsigned int> m5(10000, 10000, 2);
    OpenCLMain::instance().evaluateArithmeticFormula("1/A*(B-3)/C-(2*D+E+1)", 10000 * 10000, std::vector<unsigned int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data(), m5.data.data()});

    // EXPECT_EQ(queueToString(result), "A B + C * ");
}

