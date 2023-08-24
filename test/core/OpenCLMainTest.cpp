#include <gtest/gtest.h>
#include <deque>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/Matrix.hpp"
class OpenCLMainTest : public ::testing::Test {
public:
    Matrix<int> m1;
    Matrix<int> m2;
    Matrix<int> m3;
    Matrix<int> m4;
    Matrix<int> m5;

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
        m1 = Matrix<int>(8, 8, 1);
        m2 = Matrix<int>(8, 8, 2);
        m3 = Matrix<int>(8, 8, 3);
        m4 = Matrix<int>(8, 8, 4);
        m5 = Matrix<int>(8, 8, 5);

        matrix0 = Matrix<int>(8, 8);

        matrix1 = Matrix<int>(8, 8);
        matrix1.indexRevision(0, 7, 1);
        matrix1.shift(1, 0);
        EXPECT_EQ(matrix1.getRowShiftIndex(), 0);
        EXPECT_EQ(matrix1.getColShiftIndex(), 1);

        matrix2 = Matrix<int>(8, 8);
        matrix2.indexRevision(1, 0, 2);
        matrix2.shift(0, 1);
        EXPECT_EQ(matrix2.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix2.getColShiftIndex(), 0);

        matrix3 = Matrix<int>(8, 8);
        matrix3.indexRevision(0, 1, 3);
        matrix3.shift(-1, 0);
        EXPECT_EQ(matrix3.getRowShiftIndex(), 0);
        EXPECT_EQ(matrix3.getColShiftIndex(), 7);

        matrix4 = Matrix<int>(8, 8);
        matrix4.indexRevision(7, 0, 4);
        matrix4.shift(0, -1);
        EXPECT_EQ(matrix4.getRowShiftIndex(), 7);
        EXPECT_EQ(matrix4.getColShiftIndex(), 0);

        matrix5 = Matrix<int>(8, 8);
        matrix5.indexRevision(1, 7, 5);
        matrix5.shift(1, 1);
        EXPECT_EQ(matrix5.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix5.getColShiftIndex(), 1);

        matrix6 = Matrix<int>(8, 8);
        matrix6.indexRevision(1, 1, 6);
        matrix6.shift(-1, 1);
        EXPECT_EQ(matrix6.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix6.getColShiftIndex(), 7);

        matrix7 = Matrix<int>(8, 8);
        matrix7.indexRevision(7, 1, 7);
        matrix7.shift(-1, -1);
        EXPECT_EQ(matrix7.getRowShiftIndex(), 7);
        EXPECT_EQ(matrix7.getColShiftIndex(), 7);

        matrix8 = Matrix<int>(8, 8);
        matrix8.indexRevision(7, 7, 8);
        matrix8.shift(1, -1);
        EXPECT_EQ(matrix8.getRowShiftIndex(), 7);
        EXPECT_EQ(matrix8.getColShiftIndex(), 1);
    }

    void TearDown() override {

    }
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
    Matrix<int> result1(8, 8, 11);
    Matrix<int> result2(8, 8, 3);
    Matrix<int> result3(8, 8, 12);
    Matrix<int> result4(8, 8, 2061);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
    EXPECT_EQ(result, result4.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_AdditionShiftCase) {
    Matrix<int> result1(8, 8, 10);
    result1.indexRevision(0, 0, 11);
    Matrix<int> result2(8, 8);
    result2.indexRevision(0, 0, 3);
    Matrix<int> result3(8, 8, 10);
    result3.indexRevision(0, 0, 16);
    Matrix<int> result4(8, 8, 2046);
    result4.indexRevision(0, 0, 2076);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 + 10");
    EXPECT_EQ(result[0], 20);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", 8, 8, std::vector<int*>{matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", 8, 8, std::vector<int*>{matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", 8, 8, std::vector<int*>{matrix1.getDataData(), matrix2.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair(), matrix2.getShiftIndexPair()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", 8, 8, std::vector<int*>{matrix3.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", 8, 8, std::vector<int*>{matrix4.getDataData(), matrix5.getDataData(), matrix6.getDataData(), matrix7.getDataData(), matrix8.getDataData()}, 
        std::vector<std::pair<unsigned int, unsigned int>>{matrix4.getShiftIndexPair(), matrix5.getShiftIndexPair(), matrix6.getShiftIndexPair(), matrix7.getShiftIndexPair(), matrix8.getShiftIndexPair()});
    EXPECT_EQ(result, result4.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionBaseCase) {
    Matrix<int> result1(8, 8, -9);
    Matrix<int> result2(8, 8, -1);
    Matrix<int> result3(8, 8, -10);
    Matrix<int> result4(8, 8, 10243);
    Matrix<int> result5(8, 8, 8184);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 - 1");
    EXPECT_EQ(result[0], 7);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 - 11");
    EXPECT_EQ(result[0], -1);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - B", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 - 1 - 1 - 1 - 1");
    EXPECT_EQ(result[0], -3);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10 - A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A - B - 13 - C - D - 2032 - E", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
    EXPECT_EQ(result, result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionShiftCase) {
    Matrix<int> result1(8, 8, -10);
    result1.indexRevision(0, 0, -9);
    Matrix<int> result2(8, 8, 10);
    result2.indexRevision(0, 0, 9);
    Matrix<int> result3(8, 8, 0);
    result3.indexRevision(0, 0, -1);
    Matrix<int> result4(8, 8, -10);
    Matrix<int> result5(8, 8, -2044);
    result5.indexRevision(0, 0, -2074);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 - 10");
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10", 8, 8, std::vector<int*>{matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 - A", 8, 8, std::vector<int*>{matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - B", 8, 8, std::vector<int*>{matrix1.getDataData(), matrix2.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair(), matrix2.getShiftIndexPair()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 - 1 - 1 - 1 - 1");
    EXPECT_EQ(result[0], -3);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10 - A", 8, 8, std::vector<int*>{matrix3.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair()});
    EXPECT_EQ(result, result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 - A - B - 13 - C - D - 2032 - E", 8, 8, std::vector<int*>{matrix4.getDataData(), matrix5.getDataData(), matrix6.getDataData(), matrix7.getDataData(), matrix8.getDataData()}, 
        std::vector<std::pair<unsigned int, unsigned int>>{matrix4.getShiftIndexPair(), matrix5.getShiftIndexPair(), matrix6.getShiftIndexPair(), matrix7.getShiftIndexPair(), matrix8.getShiftIndexPair()});
    EXPECT_EQ(result, result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_MultiplicationBaseCase) {
    Matrix<int> result1(8, 8, 10);
    Matrix<int> result2(8, 8, 2);
    Matrix<int> result3(8, 8, 10);
    Matrix<int> result4(8, 8, 1000);
    Matrix<int> result5(8, 8, 1200000000);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 * 1");
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 * 11");
    EXPECT_EQ(result[0], 110);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * B", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 * 1 * 1 * 1 * 1");
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10 * A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1000 * A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1000 * A * B * 10 * C * D * 1000 * E", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
    EXPECT_EQ(result, result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_MultiplicationShiftCase) {
    Matrix<int> result1(8, 8, 0);
    result1.indexRevision(0, 0, 10);
    Matrix<int> result2(8, 8, 0);
    result2.indexRevision(0, 0, 10);
    Matrix<int> result3(8, 8, 0);
    result3.indexRevision(0, 0, 2);
    Matrix<int> result4(8, 8, 0);
    result4.indexRevision(0, 0, 90);
    Matrix<int> result5(8, 8, 0);
    result5.indexRevision(0, 0, 177515520);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("10 * 10");
    EXPECT_EQ(result[0], 100);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10", 8, 8, std::vector<int*>{matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 * A", 8, 8, std::vector<int*>{matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * B", 8, 8, std::vector<int*>{matrix1.getDataData(), matrix2.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair(), matrix2.getShiftIndexPair()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("1 * 1 * 1 * 1 * 1");
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10 * A", 8, 8, std::vector<int*>{matrix3.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair()});
    EXPECT_EQ(result, result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 * A * B * 13 * C * D * 2032 * E", 8, 8, std::vector<int*>{matrix4.getDataData(), matrix5.getDataData(), matrix6.getDataData(), matrix7.getDataData(), matrix8.getDataData()}, 
        std::vector<std::pair<unsigned int, unsigned int>>{matrix4.getShiftIndexPair(), matrix5.getShiftIndexPair(), matrix6.getShiftIndexPair(), matrix7.getShiftIndexPair(), matrix8.getShiftIndexPair()});
    EXPECT_EQ(result, result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_DivideByBaseCase) {
    Matrix<int> result1(8, 8, 0);
    Matrix<int> result2(8, 8, 2);
    Matrix<int> result3(8, 8, 10244);
    Matrix<int> result4(8, 8, 5122);
    Matrix<int> result5(8, 8, 5115);
    Matrix<int> result6(8, 8, 1707);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 / 1");
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("110 / 11");
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / 10", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", 8, 8, std::vector<int*>{m2.getDataData(), m1.getDataData()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("(7-8) / 1");
    EXPECT_EQ(result[0], -1);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (10 * A)", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / A", 8, 8, std::vector<int*>{m1.getDataData()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / (A * B)", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - 13) / (A * B)", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result5.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - A) / (B * C)", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData()});
    EXPECT_EQ(result, result6.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (2032 * B)", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("0 / (10244 + A * B - C - D)", 8, 8, std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData()});
    EXPECT_EQ(result, result1.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_DivisionShiftCase) {
    Matrix<int> result1(8, 8, 0);
    Matrix<int> result2(8, 8, 0);
    result2.indexRevision(0, 0, 2);
    Matrix<int> result3(8, 8, 0);
    result3.indexRevision(0, 0, 3414);
    Matrix<int> result4(8, 8, 0);
    result4.indexRevision(0, 0, 853);
    Matrix<int> result5(8, 8, 0);
    result5.indexRevision(0, 0, 852);
    Matrix<int> result6(8, 8, 0);
    result6.indexRevision(0, 0, 512);

    std::vector<int> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("8 / 1");
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("110 / 11");
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / 10", 8, 8, std::vector<int*>{ matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", 8, 8, std::vector<int*>{ matrix1.getDataData(), matrix2.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix1.getShiftIndexPair(), matrix2.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", 8, 8, std::vector<int*>{matrix2.getDataData(),  matrix1.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix2.getShiftIndexPair(), matrix1.getShiftIndexPair()});
    EXPECT_EQ(result, result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula<int>("(7-8) / 1");
    EXPECT_EQ(result[0], -1);
    EXPECT_EQ(result.size(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (10 * A)", 8, 8, std::vector<int*>{matrix3.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / A", 8, 8, std::vector<int*>{matrix3.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair()});
    EXPECT_EQ(result, result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / (A * B)", 8, 8, std::vector<int*>{matrix3.getDataData(), matrix4.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair(), matrix4.getShiftIndexPair()});
    EXPECT_EQ(result, result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - 13) / (A * B)", 8, 8, std::vector<int*>{matrix3.getDataData(), matrix4.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair(), matrix4.getShiftIndexPair()});
    EXPECT_EQ(result, result5.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - A) / (B * C)", 8, 8, std::vector<int*>{matrix3.getDataData(), matrix4.getDataData(), matrix5.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix3.getShiftIndexPair(), matrix4.getShiftIndexPair(), matrix5.getShiftIndexPair()});
    EXPECT_EQ(result, result6.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (2032 * B)", 8, 8, std::vector<int*>{matrix4.getDataData(), matrix5.getDataData()}, std::vector<std::pair<unsigned int, unsigned int>>{matrix4.getShiftIndexPair(), matrix5.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("0 / (10244 + A * B - C - D)", 8, 8, std::vector<int*>{matrix5.getDataData(), matrix6.getDataData(), matrix7.getDataData(), matrix8.getDataData()},
        std::vector<std::pair<unsigned int, unsigned int>>{matrix5.getShiftIndexPair(), matrix6.getShiftIndexPair(), matrix7.getShiftIndexPair(), matrix8.getShiftIndexPair()});
    EXPECT_EQ(result, result1.getShiftedData());
}

// TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_CombinedCase) {
//     Matrix<int> m1(8, 8, 1);
//     Matrix<int> m2(8, 8, 2);
//     Matrix<int> m3(8, 8, 3);
//     Matrix<int> m4(8, 8, 4);
//     Matrix<int> m5(8, 8, 5);
//     std::vector<int> result;

//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A + A + A",
//         8, 8,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result , 15);
//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A - A - A",
//         8, 8,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result , -5);
//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A * A * A",
//         8, 8,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result , 125);
//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "A / A",
//         8, 8,
//         std::vector<int*>{m5.getDataData()});
//     EXPECT_EQ(result , 1);

//     result = OpenCLMain::instance().evaluateArithmeticFormula(
//         "3 + A * (B - 4 / 2) + (C / 3) * (7 - D) + (D + 3) / E - 9",
//         8, 8,
//         std::vector<int*>{m1.getDataData(), m2.getDataData(), m3.getDataData(), m4.getDataData(), m5.getDataData()});
//     EXPECT_EQ(result , -2);
// }
