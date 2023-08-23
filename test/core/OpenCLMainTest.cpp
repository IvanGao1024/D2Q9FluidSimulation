#include <gtest/gtest.h>
#include <deque>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/Matrix.hpp"
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
    Matrix<int> m1(8, 8, 1);
    Matrix<int> m2(8, 8, 2);
    Matrix<int> m3(8, 8, 3);
    Matrix<int> m4(8, 8, 4);
    Matrix<int> m5(8, 8, 5);

    Matrix<int> result1(8, 8, 11);
    Matrix<int> result2(8, 8, 3);
    Matrix<int> result3(8, 8, 12);
    Matrix<int> result4(8, 8, 2061);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 8, 8, 64, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 8, 8, 64, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 8, 8, 64, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 8, 8, 64, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 8, 8, 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
    EXPECT_EQ(result, result4.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_AdditionShiftCase) {
    Matrix<int> m1(8, 8);
    m1.indexRevision(0, 7, 1);
    m1.shift(1, 0);
    EXPECT_EQ(m1.getRowShiftIndex(), 0);
    EXPECT_EQ(m1.getColShiftIndex(), 1);

    Matrix<int> m2(8, 8);
    m2.indexRevision(1, 0, 2);
    m2.shift(0, 1);
    EXPECT_EQ(m2.getRowShiftIndex(), 1);
    EXPECT_EQ(m2.getColShiftIndex(), 0);

    Matrix<int> m3(8, 8);
    m3.indexRevision(0, 1, 3);
    m3.shift(-1, 0);
    EXPECT_EQ(m3.getRowShiftIndex(), 0);
    EXPECT_EQ(m3.getColShiftIndex(), 7);

    Matrix<int> m4(8, 8);
    m4.indexRevision(7, 0, 4);
    m4.shift(0, -1);
    EXPECT_EQ(m4.getRowShiftIndex(), 7);
    EXPECT_EQ(m4.getColShiftIndex(), 0);

    Matrix<int> m5(8, 8);
    m5.indexRevision(1, 7, 5);
    m5.shift(1, 1);
    EXPECT_EQ(m5.getRowShiftIndex(), 1);
    EXPECT_EQ(m5.getColShiftIndex(), 1);

    Matrix<int> m6(8, 8);
    m6.indexRevision(1, 1, 6);
    m6.shift(-1, 1);
    EXPECT_EQ(m6.getRowShiftIndex(), 1);
    EXPECT_EQ(m6.getColShiftIndex(), 7);

    Matrix<int> m7(8, 8);
    m7.indexRevision(7, 1, 7);
    m7.shift(-1, -1);
    EXPECT_EQ(m7.getRowShiftIndex(), 7);
    EXPECT_EQ(m7.getColShiftIndex(), 7);

    Matrix<int> m8(8, 8);
    m8.indexRevision(7, 7, 8);
    m8.shift(1, -1);
    EXPECT_EQ(m8.getRowShiftIndex(), 7);
    EXPECT_EQ(m8.getColShiftIndex(), 1);

    Matrix<int> result1(8, 8, 10);
    result1.indexRevision(0, 0, 11);
    Matrix<int> result2(8, 8);
    result2.indexRevision(0, 0, 3);
    // Matrix<int> result3(8, 8, 10);
    // result3.rowRevision(0, 12);
    // Matrix<int> result4(8, 8, 2046);
    // result4.rowRevision(0, 2061);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 8, 8, 64, std::vector<int*>{m1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{m1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 8, 8, 64, std::vector<int*>{m1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{m1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 8, 8, 64, std::vector<int*>{m1.getDataData(), m2.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{m1.getShiftIndexPair(), m2.getShiftIndexPair()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    // result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 8, 8, 64, std::vector<int*>{m1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{m1.getShiftIndexPair()});
    // EXPECT_EQ(result, result3.getShiftedData());
    // result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 8, 8, 64, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()}, 
    //     std::vector<std::pair<unsigned int, unsigned int>>{m1.getShiftIndexPair(), m2.getShiftIndexPair(), m3.getShiftIndexPair(), m4.getShiftIndexPair(), m5.getShiftIndexPair()});
    // EXPECT_EQ(result, result4.getShiftedData());
}

// TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionBaseCase) {
//     Matrix<int> m1(8, 8, 1);
//     Matrix<int> m2(8, 8, 2);
//     Matrix<int> m3(8, 8, 3);
//     Matrix<int> m4(8, 8, 4);
//     Matrix<int> m5(8, 8, 5);
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
//     Matrix<int> m1(8, 8, 1);
//     Matrix<int> m2(8, 8, 2);
//     Matrix<int> m3(8, 8, 3);
//     Matrix<int> m4(8, 8, 4);
//     Matrix<int> m5(8, 8, 5);
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
//     Matrix<int> m1(8, 8, 1);
//     Matrix<int> m2(8, 8, 2);
//     Matrix<int> m3(8, 8, 3);
//     Matrix<int> m4(8, 8, 4);
//     Matrix<int> m5(8, 8, 5);
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
//     Matrix<int> m1(8, 8, 1);
//     Matrix<int> m2(8, 8, 2);
//     Matrix<int> m3(8, 8, 3);
//     Matrix<int> m4(8, 8, 4);
//     Matrix<int> m5(8, 8, 5);
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
