#include <gtest/gtest.h>
#include <deque>
#include "../../src/core/OpenCLMain.hpp"
#include "../../src/core/Matrix.hpp"
class OpenCLMainTest : public ::testing::Test {
public:
    Matrix<double> m1;
    Matrix<double> m2;
    Matrix<double> m3;
    Matrix<double> m4;
    Matrix<double> m5;

    Matrix<double> matrix0;
    Matrix<double> matrix1;
    Matrix<double> matrix2;
    Matrix<double> matrix3;
    Matrix<double> matrix4;
    Matrix<double> matrix5;
    Matrix<double> matrix6;
    Matrix<double> matrix7;
    Matrix<double> matrix8;

protected:
    void SetUp() override {
        m1 = Matrix<double>(8, 8, 1);
        m2 = Matrix<double>(8, 8, 2);
        m3 = Matrix<double>(8, 8, 3);
        m4 = Matrix<double>(8, 8, 4);
        m5 = Matrix<double>(8, 8, 5);

        matrix0 = Matrix<double>(8, 8);

        matrix1 = Matrix<double>(8, 8);
        matrix1.indexRevision(0, 7, 1);
        matrix1.shift(1, 0);
        EXPECT_EQ(matrix1.getRowShiftIndex(), 0);
        EXPECT_EQ(matrix1.getColShiftIndex(), 1);

        matrix2 = Matrix<double>(8, 8);
        matrix2.indexRevision(1, 0, 2);
        matrix2.shift(0, 1);
        EXPECT_EQ(matrix2.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix2.getColShiftIndex(), 0);

        matrix3 = Matrix<double>(8, 8);
        matrix3.indexRevision(0, 1, 3);
        matrix3.shift(-1, 0);
        EXPECT_EQ(matrix3.getRowShiftIndex(), 0);
        EXPECT_EQ(matrix3.getColShiftIndex(), 7);

        matrix4 = Matrix<double>(8, 8);
        matrix4.indexRevision(7, 0, 4);
        matrix4.shift(0, -1);
        EXPECT_EQ(matrix4.getRowShiftIndex(), 7);
        EXPECT_EQ(matrix4.getColShiftIndex(), 0);

        matrix5 = Matrix<double>(8, 8);
        matrix5.indexRevision(1, 7, 5);
        matrix5.shift(1, 1);
        EXPECT_EQ(matrix5.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix5.getColShiftIndex(), 1);

        matrix6 = Matrix<double>(8, 8);
        matrix6.indexRevision(1, 1, 6);
        matrix6.shift(-1, 1);
        EXPECT_EQ(matrix6.getRowShiftIndex(), 1);
        EXPECT_EQ(matrix6.getColShiftIndex(), 7);

        matrix7 = Matrix<double>(8, 8);
        matrix7.indexRevision(7, 1, 7);
        matrix7.shift(-1, -1);
        EXPECT_EQ(matrix7.getRowShiftIndex(), 7);
        EXPECT_EQ(matrix7.getColShiftIndex(), 7);

        matrix8 = Matrix<double>(8, 8);
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
    Matrix<double> result1(8, 8, 11);
    Matrix<double> result2(8, 8, 3);
    Matrix<double> result3(8, 8, 12);
    Matrix<double> result4(8, 8, 2061);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + 10");
    EXPECT_EQ(result.getValue(0), 20);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result.getValue(0), 5);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", std::vector<Matrix<double>*>{&m1, &m2, &m3, &m4, &m5});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_AdditionShiftCase) {
    Matrix<double> result1(8, 8, 10);
    result1.indexRevision(0, 0, 11);
    Matrix<double> result2(8, 8);
    result2.indexRevision(0, 0, 3);
    Matrix<double> result3(8, 8, 10);
    result3.indexRevision(0, 0, 16);
    Matrix<double> result4(8, 8, 2046);
    result4.indexRevision(0, 0, 2076);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + 10");
    EXPECT_EQ(result.getValue(0), 20);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 + A", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + B", std::vector<Matrix<double>*>{&matrix1, &matrix2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + 1 + 1 + 1 + 1");
    EXPECT_EQ(result.getValue(0), 5);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A + 10 + A", std::vector<Matrix<double>*>{&matrix3});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 + A + B + 13 + C + D + 2032 + E", std::vector<Matrix<double>*>{&matrix4, &matrix5, &matrix6, &matrix7, &matrix8});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionBaseCase) {
    Matrix<double> result1(8, 8, -9);
    Matrix<double> result2(8, 8, -1);
    Matrix<double> result3(8, 8, -10);
    Matrix<double> result4(8, 8, 10243);
    Matrix<double> result5(8, 8, 8184);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("8 - 1");
    EXPECT_EQ(result.getValue(0), 7);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 - 11");
    EXPECT_EQ(result.getValue(0), -1);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - B", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 - 1 - 1 - 1 - 1");
    EXPECT_EQ(result.getValue(0), -3);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10 - A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 - A - B - 13 - C - D - 2032 - E", std::vector<Matrix<double>*>{&m1, &m2, &m3, &m4, &m5});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_SubtractionShiftCase) {
    Matrix<double> result1(8, 8, -10);
    result1.indexRevision(0, 0, -9);
    Matrix<double> result2(8, 8, 10);
    result2.indexRevision(0, 0, 9);
    Matrix<double> result3(8, 8, 0);
    result3.indexRevision(0, 0, -1);
    Matrix<double> result4(8, 8, -10);
    Matrix<double> result5(8, 8, -2044);
    result5.indexRevision(0, 0, -2074);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 - 10");
    EXPECT_EQ(result.getValue(0), 0);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 - A", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - B", std::vector<Matrix<double>*>{&matrix1, &matrix2});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 - 1 - 1 - 1 - 1");
    EXPECT_EQ(result.getValue(0), -3);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A - 10 - A", std::vector<Matrix<double>*>{&matrix3});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 - A - B - 13 - C - D - 2032 - E", std::vector<Matrix<double>*>{&matrix4, &matrix5, &matrix6, &matrix7, &matrix8});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_MultiplicationBaseCase) {
    Matrix<double> result1(8, 8, 10);
    Matrix<double> result2(8, 8, 2);
    Matrix<double> result3(8, 8, 10);
    Matrix<double> result4(8, 8, 1000);
    Matrix<double> result5(8, 8, 1200000000);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("8 * 1");
    EXPECT_EQ(result.getValue(0), 8);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 * 11");
    EXPECT_EQ(result.getValue(0), 110);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * B", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 * 1 * 1 * 1 * 1");
    EXPECT_EQ(result.getValue(0), 1);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10 * A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1000 * A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1000 * A * B * 10 * C * D * 1000 * E", std::vector<Matrix<double>*>{&m1, &m2, &m3, &m4, &m5});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_MultiplicationShiftCase) {
    Matrix<double> result1(8, 8, 0);
    result1.indexRevision(0, 0, 10);
    Matrix<double> result2(8, 8, 0);
    result2.indexRevision(0, 0, 10);
    Matrix<double> result3(8, 8, 0);
    result3.indexRevision(0, 0, 2);
    Matrix<double> result4(8, 8, 0);
    result4.indexRevision(0, 0, 90);
    Matrix<double> result5(8, 8, 0);
    result5.indexRevision(0, 0, 177515520);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 * 10");
    EXPECT_EQ(result.getValue(0), 100);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10 * A", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * B", std::vector<Matrix<double>*>{&matrix1, &matrix2});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 * 1 * 1 * 1 * 1");
    EXPECT_EQ(result.getValue(0), 1);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A * 10 * A", std::vector<Matrix<double>*>{&matrix3});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("1 * A * B * 13 * C * D * 2032 * E", std::vector<Matrix<double>*>{&matrix4, &matrix5, &matrix6, &matrix7, &matrix8});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_DivideByBaseCase) {
    Matrix<double> result1(8, 8, 0.1);
    Matrix<double> result2(8, 8, 0.5);
    Matrix<double> result3(8, 8, 2);
    Matrix<double> result4(8, 8, 0.1);
    Matrix<double> result5(8, 8, 10244);
    Matrix<double> result6(8, 8, 5122);
    Matrix<double> result7(8, 8, 5115.5);
    Matrix<double> result8(8, 8, 1707.1666666666667);
    Matrix<double> result9(8, 8, 0.00024606299212598425);
    Matrix<double> result10(8, 8, 0);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("8 / 1");
    EXPECT_EQ(result.getValue(0), 8);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("110 / 11");
    EXPECT_EQ(result.getValue(0), 10);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / 10", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", std::vector<Matrix<double>*>{&m2, &m1});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(7-8) / 1");
    EXPECT_EQ(result.getValue(0), -1);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (10 * A)", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / (A * B)", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result6.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - 13) / (A * B)", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result7.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - A) / (B * C)", std::vector<Matrix<double>*>{&m1, &m2, &m3});
    EXPECT_EQ(result.getShiftedData(), result8.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (2032 * B)", std::vector<Matrix<double>*>{&m1, &m2});
    EXPECT_EQ(result.getShiftedData(), result9.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("0 / (10244 + A * B - C - D)", std::vector<Matrix<double>*>{&m1, &m2, &m3, &m4});
    EXPECT_EQ(result.getShiftedData(), result10.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_DivisionShiftCase) {
    Matrix<double> result1(8, 8, 0);
    result1.indexRevision(0, 0, 0.1);
    Matrix<double> result2(8, 8, 0);
    result2.indexRevision(0, 0, 0.5);
    Matrix<double> result3(8, 8, 0);
    result3.indexRevision(0, 0, 2);
    Matrix<double> result4(8, 8, 0);
    result4.indexRevision(0, 0, 0.1);
    Matrix<double> result5(8, 8, 0);
    result5.indexRevision(0, 0, 3414.6666666666665);
    Matrix<double> result6(8, 8, 0);
    result6.indexRevision(0, 0, 853.66666666666663);
    Matrix<double> result7(8, 8, 0);
    result7.indexRevision(0, 0, 852.58333333333337);
    Matrix<double> result8(8, 8, 0);
    result8.indexRevision(0, 0, 512.05);
    Matrix<double> result9(8, 8, 0);
    result9.indexRevision(0, 0, 0.0003937007874015748);
    Matrix<double> result10(8, 8, 0);
    result10.indexRevision(0, 0, 0);


    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula("8 / 1");
    EXPECT_EQ(result.getValue(0), 8);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("110 / 11");
    EXPECT_EQ(result.getValue(0), 10);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / 10", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", std::vector<Matrix<double>*>{&matrix1, &matrix2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / B", std::vector<Matrix<double>*>{&matrix2, &matrix1});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(7-8) / 1");
    EXPECT_EQ(result.getValue(0), -1);
    EXPECT_EQ(result.getLength(), 1);
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (10 * A)", std::vector<Matrix<double>*>{&matrix3});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / A", std::vector<Matrix<double>*>{&matrix3});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("10244 / (A * B)", std::vector<Matrix<double>*>{&matrix3, &matrix4});
    EXPECT_EQ(result.getShiftedData(), result6.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - 13) / (A * B)", std::vector<Matrix<double>*>{&matrix3, &matrix4});
    EXPECT_EQ(result.getShiftedData(), result7.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("(10244 - A) / (B * C)", std::vector<Matrix<double>*>{&matrix3, &matrix4, &matrix5});
    EXPECT_EQ(result.getShiftedData(), result8.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("A / (2032 * B)", std::vector<Matrix<double>*>{&matrix4, &matrix5});
    EXPECT_EQ(result.getShiftedData(), result9.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula("0 / (10244 + A * B - C - D)", std::vector<Matrix<double>*>{&matrix5, &matrix6, &matrix7, &matrix8});
    EXPECT_EQ(result.getShiftedData(), result10.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_CombinedBaseCase) {
    Matrix<double> result1(8, 8, 3);
    Matrix<double> result2(8, 8, -2);
    Matrix<double> result3(8, 8, 27);
    Matrix<double> result4(8, 8, 1);
    Matrix<double> result5(8, 8, -1.5999999999999996);
    Matrix<double> result6(8, 8, 0.8);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A + A + A", std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A - A - A", std::vector<Matrix<double>*>{&m2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A * A * A", std::vector<Matrix<double>*>{&m3});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A / A", std::vector<Matrix<double>*>{&m4});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "3 + A * (B - 4 / 2) + (C / 3) * (7 - D) + (D + 3) / E - 9", 
        std::vector<Matrix<double>*>{&m1, &m2, &m3, &m4, &m5});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
    m1.fill(0.25);
    m1.print();
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "1 / ((A * 3) + 0.5)", 
        std::vector<Matrix<double>*>{&m1});
    EXPECT_EQ(result.getShiftedData(), result6.getShiftedData());
}

TEST_F(OpenCLMainTest, EvaluateArithmeticFormulaTest_CombinedShiftCase) {
    Matrix<double> result1(8, 8, 0);
    result1.indexRevision(0, 0, 3);
    Matrix<double> result2(8, 8, 0);
    result2.indexRevision(0, 0, -2);
    Matrix<double> result3(8, 8, 0);
    result3.indexRevision(0, 0, 27);
    Matrix<double> result4(8, 8, 0);
    result4.indexRevision(0, 0, 1);
    Matrix<double> result5(8, 8, -6);
    result5.indexRevision(0, 0, 11.666666666666668);

    Matrix<double> result;
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A + A + A", std::vector<Matrix<double>*>{&matrix1});
    EXPECT_EQ(result.getShiftedData(), result1.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A - A - A", std::vector<Matrix<double>*>{&matrix2});
    EXPECT_EQ(result.getShiftedData(), result2.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A * A * A", std::vector<Matrix<double>*>{&matrix3});
    EXPECT_EQ(result.getShiftedData(), result3.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "A / A", std::vector<Matrix<double>*>{&matrix4});
    EXPECT_EQ(result.getShiftedData(), result4.getShiftedData());
    result = OpenCLMain::instance().evaluateArithmeticFormula(
        "3 + A * (B - 4 / 2) + (C / 3) * (7 - D) + (D + 3) / E - 9", 
        std::vector<Matrix<double>*>{&matrix5, &matrix6, &matrix7, &matrix8, &matrix0});
    EXPECT_EQ(result.getShiftedData(), result5.getShiftedData());
}