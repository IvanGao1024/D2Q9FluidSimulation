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

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_AdditionBaseCase) {
    CartesianMatrix<int> m1(5, 5, 1);
    CartesianMatrix<int> m2(5, 5, 2);
    CartesianMatrix<int> m3(5, 5, 3);
    CartesianMatrix<int> m4(5, 5, 4);
    CartesianMatrix<int> m5(5, 5, 5);
    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 11);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 11);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 25, std::vector<int*>{m1.data.data(), m2.data.data()});
    EXPECT_EQ(result[0], 3);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 12);
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 25, std::vector<int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data(), m5.data.data()});
    EXPECT_EQ(result[0], 2061);
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionBaseCase) {
    CartesianMatrix<int> m1(5, 5, 1);
    CartesianMatrix<int> m2(5, 5, 2);
    CartesianMatrix<int> m3(5, 5, 3);
    CartesianMatrix<int> m4(5, 5, 4);
    CartesianMatrix<int> m5(5, 5, 5);
    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("5 - 1");
    EXPECT_EQ(result[0], 4);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 - 11");
    EXPECT_EQ(result[0], -1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], -9);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - B", 25, std::vector<int*>{m1.data.data(), m2.data.data()});
    EXPECT_EQ(result[0], -1);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 - 1 - 1 - 1 - 1");
    EXPECT_EQ(result[0], -3);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10 - A", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], -10);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 10243);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A - B - 13 - C - D - 2032 - E", 25, std::vector<int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data(), m5.data.data()});
    EXPECT_EQ(result[0], 8184);
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_MultiplicationBaseCase) {
    CartesianMatrix<int> m1(5, 5, 1);
    CartesianMatrix<int> m2(5, 5, 2);
    CartesianMatrix<int> m3(5, 5, 3);
    CartesianMatrix<int> m4(5, 5, 4);
    CartesianMatrix<int> m5(5, 5, 5);
    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("5 * 1");
    EXPECT_EQ(result[0], 5);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 * 11");
    EXPECT_EQ(result[0], 110);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 10);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * B", 25, std::vector<int*>{m1.data.data(), m2.data.data()});
    EXPECT_EQ(result[0], 2);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 * (4-5)");
    EXPECT_EQ(result[0], -1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10 * A", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 10);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 * A", 25, std::vector<int*>{m1.data.data()});
    EXPECT_EQ(result[0], 10244);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A * B", 25, std::vector<int*>{m1.data.data(), m2.data.data()});
    EXPECT_EQ(result[0], 10242);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A * B + 13", 25, std::vector<int*>{m1.data.data(), m2.data.data()});
    EXPECT_EQ(result[0], 10255);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A * B + 13 * C", 25, std::vector<int*>{m1.data.data(), m2.data.data(), m3.data.data()});
    EXPECT_EQ(result[0], 10281);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 2032 * B", 25, std::vector<int*>{m1.data.data(), m2.data.data()});
    EXPECT_EQ(result[0], -4063);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A * B + 13 * C + D - 2032", 25, std::vector<int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data()});
    EXPECT_EQ(result[0], 8253);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A * B + 13 * C + D - 2032 * E", 25, std::vector<int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data(), m5.data.data()});
    EXPECT_EQ(result[0], 125);
    result = OpenCLMain::instance().evaluateArithmeticFormula("0 - 10244 - A * B + 13 * C + D - 2032 * E", 25, std::vector<int*>{m1.data.data(), m2.data.data(), m3.data.data(), m4.data.data(), m5.data.data()});
    EXPECT_EQ(result[0], -20363);
}