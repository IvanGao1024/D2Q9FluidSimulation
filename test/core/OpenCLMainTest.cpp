#include <gtest/gtest.h>
#include <deque>
#include "../../src/core/OpenCLMain.hpp"
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
    OpenCLMain::instance().evaluateArithmeticFormula("1/A*(B-3)/C-(2*D+E+1)", 3, 4, 6, 3, 1);
    // EXPECT_EQ(queueToString(result), "A B + C * ");
}

