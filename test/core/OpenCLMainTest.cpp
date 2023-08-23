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

TEST_F(OpenCLMainTest, ShuntingYardTest_Complex) {
    std::queue<std::string> result = OpenCLMain::instance().enqueueArithmeticFormula(
        "3 + A * (B - 4 / 2) + (C / 3) * (7 - E) + (E + 3) / D  - 9");
    EXPECT_EQ(queueToString(result), "3 A B 4 2 / - * + C 3 / 7 E - * + E 3 + D / + 9 - ");
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_AdditionBaseCase) {
    CartesianMatrix<int> m1(8, 8, 1);
    CartesianMatrix<int> m2(8, 8, 2);
    CartesianMatrix<int> m3(8, 8, 3);
    CartesianMatrix<int> m4(8, 8, 4);
    CartesianMatrix<int> m5(8, 8, 5);
    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 64, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result[0], 11);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 64, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result[0], 11);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result[0], 3);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 64, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result[0], 12);
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
    EXPECT_EQ(result[0], 2061);
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_AdditionShiftCase) {
    CartesianMatrix<int> m1(8, 8);
    m1.rowRevision(0, 1);
    EXPECT_EQ(m1.getShiftIndex(), 0);
    CartesianMatrix<int> m2(8, 8);
    m2.rowRevision(1, 2);
    m2.shiftUp(1);
    EXPECT_EQ(m2.getShiftIndex(), 8);
    CartesianMatrix<int> m3(8, 8);
    m3.rowRevision(2, 3);
    m3.shiftUp(2);
    EXPECT_EQ(m3.getShiftIndex(), 16);
    CartesianMatrix<int> m4(8, 8);
    m4.rowRevision(3, 4);
    m4.shiftUp(3);
    EXPECT_EQ(m4.getShiftIndex(), 24);
    CartesianMatrix<int> m5(8, 8);
    m5.rowRevision(4, 5);
    m5.shiftUp(4);
    EXPECT_EQ(m5.getShiftIndex(), 32);

    CartesianMatrix<int> result1(8, 8, 10);
    result1.rowRevision(0, 11);
    CartesianMatrix<int> result2(8, 8, 0);
    result2.rowRevision(0, 3);
    CartesianMatrix<int> result3(8, 8, 10);
    result3.rowRevision(0, 12);
    CartesianMatrix<int> result4(8, 8, 2046);
    result4.rowRevision(0, 2061);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 64,std::vector<int*>{m1.getDataData()}, std::vector<unsigned int>{m1.getShiftIndex()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 64, std::vector<int*>{m1.getDataData()}, std::vector<unsigned int>{m1.getShiftIndex()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()}, std::vector<unsigned int>{m1.getShiftIndex(), m2.getShiftIndex()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 64, std::vector<int*>{m1.getDataData()}, std::vector<unsigned int>{m1.getShiftIndex()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()}, 
        std::vector<unsigned int>{m1.getShiftIndex(), m2.getShiftIndex(), m3.getShiftIndex(), m4.getShiftIndex(), m5.getShiftIndex()});
    EXPECT_EQ(result, result4.getShiftedData());
}

// TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionBaseCase) {
//     CartesianMatrix<int> m1(8, 8, 1);
//     CartesianMatrix<int> m2(8, 8, 2);
//     CartesianMatrix<int> m3(8, 8, 3);
//     CartesianMatrix<int> m4(8, 8, 4);
//     CartesianMatrix<int> m5(8, 8, 5);
//     std::vector<int> result;
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 - 1");
//     EXPECT_EQ(result[0], 7);
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 - 11");
//     EXPECT_EQ(result[0], -1);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], -9);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A - B", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
//     EXPECT_EQ(result[0], -1);
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 - 1 - 1 - 1 - 1");
//     EXPECT_EQ(result[0], -3);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10 - A", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], -10);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 10243);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A - B - 13 - C - D - 2032 - E", 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
//     EXPECT_EQ(result[0], 8184);
// }

// TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_MultiplicationBaseCase) {
//     CartesianMatrix<int> m1(8, 8, 1);
//     CartesianMatrix<int> m2(8, 8, 2);
//     CartesianMatrix<int> m3(8, 8, 3);
//     CartesianMatrix<int> m4(8, 8, 4);
//     CartesianMatrix<int> m5(8, 8, 5);
//     std::vector<int> result;
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 * 1");
//     EXPECT_EQ(result[0], 8);
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 * 11");
//     EXPECT_EQ(result[0], 110);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 10);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A * B", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
//     EXPECT_EQ(result[0], 2);
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 * 1 * 1 * 1 * 1");
//     EXPECT_EQ(result[0], 1);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10 * A", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 10);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("1000 * A", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 1000);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("1000 * A * B * 10 * C * D * 1000 * E", 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
//     EXPECT_EQ(result[0], 1200000000);
// }

// TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_DivideByBaseCase) {
//     CartesianMatrix<int> m1(8, 8, 1);
//     CartesianMatrix<int> m2(8, 8, 2);
//     CartesianMatrix<int> m3(8, 8, 3);
//     CartesianMatrix<int> m4(8, 8, 4);
//     CartesianMatrix<int> m5(8, 8, 5);
//     std::vector<int> result;

//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 / 1");
//     EXPECT_EQ(result[0], 8);
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("110 / 11");
//     EXPECT_EQ(result[0], 10);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A / 10", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 0); // integer division: 1/10 = 0
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
//     EXPECT_EQ(result[0], 0);
//     result = OpenCLMain::instance().evaluateArithmeticFormula<int>("(7-8) / 1");
//     EXPECT_EQ(result[0], -1);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A / (10 * A)", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 0);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / A", 64, std::vector<int*>{m1.getDataData()});
//     EXPECT_EQ(result[0], 10244);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / (A * B)", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
//     EXPECT_EQ(result[0], 5122);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - 13) / (A * B)", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
//     EXPECT_EQ(result[0], 5115);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - A) / (B * C)", 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData()});
//     EXPECT_EQ(result[0], 1707);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("A / (2032 * B)", 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
//     EXPECT_EQ(result[0], 0);
//     result = OpenCLMain::instance().evaluateArithmeticFormula("0 / (10244 + A * B - C - D)", 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData()});
//     EXPECT_EQ(result[0], 0);
// }

// TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_CombinedCase) {
//     CartesianMatrix<int> m1(8, 8, 1);
//     CartesianMatrix<int> m2(8, 8, 2);
//     CartesianMatrix<int> m3(8, 8, 3);
//     CartesianMatrix<int> m4(8, 8, 4);
//     CartesianMatrix<int> m5(8, 8, 5);
//     std::vector<int> result;

//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A + A + A",
//         64,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result[0], 15);
//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A - A - A",
//         64,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result[0], -5);
//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A * A * A",
//         64,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result[0], 125);
//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A / A",
//         64,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result[0], 1);

//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "3 + A * (B - 4 / 2) + (C / 3) * (7 - D) + (D + 3) / E - 9",
//         64,
//         std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
//     EXPECT_EQ(result[0], -2);
// }
