#include "LatticeBoltzmannMethodD2Q9.h"

#include <omp.h>
#include <cassert>

#include "OpenCLMain.hpp"

LatticeBoltzmannMethodD2Q9::LatticeBoltzmannMethodD2Q9(unsigned int        height,
													   unsigned int        width,
													   Boundary            top,
													   Boundary            bottom,
													   Boundary            left,
													   Boundary            right,
													   std::vector<double> kinematicViscosityArray,
													   std::vector<double> diffusionCoefficientArray,
													   std::vector<double> initialDensityArray,
													   std::vector<double> initialTemperatureArray)
{
	mHeight = height + 1;
	mWidth  = width + 1;
	mLength = mHeight * mWidth;
	mTop    = top;
	mBottom = bottom;
	mLeft   = left;
	mRight  = right;
	// mEntities = entities;

	mKinematicViscosityRevised   = true;
	mDiffusionCoefficientRevised = true;
	mKinematicViscosity          = Matrix<double>(mWidth, mHeight, kinematicViscosityArray);
	mDiffusionCoefficient        = Matrix<double>(mWidth, mHeight, diffusionCoefficientArray);

	if(initialDensityArray.empty()) {
		initialDensityArray.resize(mLength);
	}
	if(initialTemperatureArray.empty()) {
		initialTemperatureArray.resize(mLength);
	}

#pragma omp parallel sections
	{
#pragma omp section
		{mDensity[0] = Matrix<double>(mWidth, mHeight, initialDensityArray, 4 / 9.0);
}
#pragma omp section
{
	mDensity[1] = Matrix<double>(mWidth, mHeight, initialDensityArray, 1 / 9.0);
}
#pragma omp section
{
	mDensity[5] = Matrix<double>(mWidth, mHeight, initialDensityArray, 1 / 36.0);
}
#pragma omp section
{
	mTemperature[0] = Matrix<double>(mWidth, mHeight, initialTemperatureArray, 4 / 9.0);
}
#pragma omp section
{
	mTemperature[1] = Matrix<double>(mWidth, mHeight, initialTemperatureArray, 1 / 9.0);
}
#pragma omp section
{
	mTemperature[5] = Matrix<double>(mWidth, mHeight, initialTemperatureArray, 1 / 36.0);
}
}

#pragma omp parallel sections
{
#pragma omp section
	{
		mDensity[2] = mDensity[1];
	}
#pragma omp section
	{
		mDensity[3] = mDensity[1];
	}
#pragma omp section
	{
		mDensity[4] = mDensity[1];
	}
#pragma omp section
	{
		mDensity[6] = mDensity[5];
	}
#pragma omp section
	{
		mDensity[7] = mDensity[5];
	}
#pragma omp section
	{
		mDensity[8] = mDensity[5];
	}
#pragma omp section
	{
		mTemperature[2] = mTemperature[1];
	}
#pragma omp section
	{
		mTemperature[3] = mTemperature[1];
	}
#pragma omp section
	{
		mTemperature[4] = mTemperature[1];
	}
#pragma omp section
	{
		mTemperature[6] = mTemperature[5];
	}
#pragma omp section
	{
		mTemperature[7] = mTemperature[5];
	}
#pragma omp section
	{
		mTemperature[8] = mTemperature[5];
	}
}
OpenCLMain::instance();
}

void LatticeBoltzmannMethodD2Q9::step(bool saveImage)
{
	updateVelocityMatrix();
	collision();
	streaming();

	if(saveImage) {
		buildResultingDensityMatrix();
		buildResultingTemperatureMatrix();
		std::cout << "mResultingDensityMatrix\n";
		mResultingDensityMatrix.print();
		std::cout << "mResultingTemperatureMatrix\n";
		mResultingTemperatureMatrix.print();
	}
}

void LatticeBoltzmannMethodD2Q9::collision()
{
	buildResultingDensityMatrix();
	buildResultingTemperatureMatrix();
	if(mKinematicViscosityRevised) {
		mOmega_m = OpenCLMain::instance().evaluateArithmeticFormula("1 / ((A * 3) + 0.5)",
																	std::vector<Matrix<double>*>{&mKinematicViscosity});
	}
	if(mDiffusionCoefficientRevised) {
		mOmega_s =
			OpenCLMain::instance().evaluateArithmeticFormula("1 / ((A * 3) + 0.5)",
															 std::vector<Matrix<double>*>{&mDiffusionCoefficient});
	}

	mTemperature[0] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * 1",
		std::vector<Matrix<double>*>{&mTemperature[0], &mOmega_s, &mResultingTemperatureMatrix});
	mTemperature[1] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 + 3 * D)",
		std::vector<Matrix<double>*>{&mTemperature[1], &mOmega_s, &mResultingTemperatureMatrix, &mVelocityU});
	mTemperature[2] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 + 3 * D)",
		std::vector<Matrix<double>*>{&mTemperature[2], &mOmega_s, &mResultingTemperatureMatrix, &mVelocityV});
	mTemperature[3] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 - 3 * D)",
		std::vector<Matrix<double>*>{&mTemperature[3], &mOmega_s, &mResultingTemperatureMatrix, &mVelocityU});
	mTemperature[4] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 - 3 * D)",
		std::vector<Matrix<double>*>{&mTemperature[4], &mOmega_s, &mResultingTemperatureMatrix, &mVelocityV});
	mTemperature[5] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 + 3 * D + 3 * E)",
														 std::vector<Matrix<double>*>{&mTemperature[5],
																					  &mOmega_s,
																					  &mResultingTemperatureMatrix,
																					  &mVelocityU,
																					  &mVelocityV});
	mTemperature[6] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 - 3 * D + 3 * E)",
														 std::vector<Matrix<double>*>{&mTemperature[6],
																					  &mOmega_s,
																					  &mResultingTemperatureMatrix,
																					  &mVelocityU,
																					  &mVelocityV});
	mTemperature[7] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 - 3 * D - 3 * E)",
														 std::vector<Matrix<double>*>{&mTemperature[7],
																					  &mOmega_s,
																					  &mResultingTemperatureMatrix,
																					  &mVelocityU,
																					  &mVelocityV});
	mTemperature[8] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 + 3 * D - 3 * E)",
														 std::vector<Matrix<double>*>{&mTemperature[8],
																					  &mOmega_s,
																					  &mResultingTemperatureMatrix,
																					  &mVelocityU,
																					  &mVelocityV});
	mResultU2 = OpenCLMain::instance().evaluateArithmeticFormula("A * A", std::vector<Matrix<double>*>{&mVelocityU});
	mResultV2 = OpenCLMain::instance().evaluateArithmeticFormula("A * A", std::vector<Matrix<double>*>{&mVelocityV});
	mResultUV2 =
		OpenCLMain::instance().evaluateArithmeticFormula("A + B", std::vector<Matrix<double>*>{&mResultU2, &mResultV2});
	mDensity[0] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 - 1.5 * D)",
		std::vector<Matrix<double>*>{&mDensity[0], &mOmega_m, &mResultingDensityMatrix, &mResultUV2});
	mDensity[1] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 + 3 * D + 4.5 * E - 1.5 * F)",
		std::vector<Matrix<double>*>{&mDensity[1],
									 &mOmega_m,
									 &mResultingDensityMatrix,
									 &mVelocityU,
									 &mResultU2,
									 &mResultUV2});
	mDensity[2] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 + 3 * D + 4.5 * E - 1.5 * F)",
		std::vector<Matrix<double>*>{&mDensity[2],
									 &mOmega_m,
									 &mResultingDensityMatrix,
									 &mVelocityV,
									 &mResultV2,
									 &mResultUV2});
	mDensity[3] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 - 3 * D + 4.5 * E - 1.5 * F)",
		std::vector<Matrix<double>*>{&mDensity[3],
									 &mOmega_m,
									 &mResultingDensityMatrix,
									 &mVelocityU,
									 &mResultU2,
									 &mResultUV2});
	mDensity[4] = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (1 - B) + B * (4/9) * C * (1 - 3 * D + 4.5 * E - 1.5 * F)",
		std::vector<Matrix<double>*>{&mDensity[4],
									 &mOmega_m,
									 &mResultingDensityMatrix,
									 &mVelocityV,
									 &mResultV2,
									 &mResultUV2});
	mDensity[5] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 + 3 * D + 3 * E + 3 * F)",
														 std::vector<Matrix<double>*>{&mDensity[5],
																					  &mOmega_m,
																					  &mResultingDensityMatrix,
																					  &mVelocityU,
																					  &mVelocityV,
																					  &mResultUV2});
	mDensity[6] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 - 3 * D + 3 * E + 3 * F)",
														 std::vector<Matrix<double>*>{&mDensity[6],
																					  &mOmega_m,
																					  &mResultingDensityMatrix,
																					  &mVelocityU,
																					  &mVelocityV,
																					  &mResultUV2});
	mDensity[7] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 - 3 * D - 3 * E + 3 * F)",
														 std::vector<Matrix<double>*>{&mDensity[7],
																					  &mOmega_m,
																					  &mResultingDensityMatrix,
																					  &mVelocityU,
																					  &mVelocityV,
																					  &mResultUV2});
	mDensity[8] =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (1 - B) + B * (4/9) * C * (1 + 3 * D - 3 * E + 3 * F)",
														 std::vector<Matrix<double>*>{&mDensity[8],
																					  &mOmega_m,
																					  &mResultingDensityMatrix,
																					  &mVelocityU,
																					  &mVelocityV,
																					  &mResultUV2});
}

void LatticeBoltzmannMethodD2Q9::streaming()
{
	mDensity[0].shift(0, 0);
	mDensity[1].shift(1, 0);
	mDensity[2].shift(0, 1);
	mDensity[3].shift(-1, 0);
	mDensity[4].shift(0, -1);
	mDensity[5].shift(1, 1);
	mDensity[6].shift(-1, 1);
	mDensity[7].shift(-1, -1);
	mDensity[8].shift(1, -1);

	mTemperature[0].shift(0, 0);
	mTemperature[1].shift(1, 0);
	mTemperature[2].shift(0, 1);
	mTemperature[3].shift(-1, 0);
	mTemperature[4].shift(0, -1);
	mTemperature[5].shift(1, 1);
	mTemperature[6].shift(-1, 1);
	mTemperature[7].shift(-1, -1);
	mTemperature[8].shift(1, -1);

	switch(mTop.boundary) {
	case 0:
		mDensity[4].topAdiabatic();
		mTemperature[4].topAdiabatic();
		mDensity[7].topAdiabatic();
		mTemperature[7].topAdiabatic();
		mDensity[8].topAdiabatic();
		mTemperature[8].topAdiabatic();
		break;
	case 1:
		mDensity[4].topDirichlet((2 / 9.0) * mTop.parameter1, mDensity[2]);
		mTemperature[4].topDirichlet((2 / 9.0) * mTop.parameter1, mTemperature[2]);
		mDensity[7].topDirichlet((2 / 36.0) * mTop.parameter1, mDensity[5]);
		mTemperature[7].topDirichlet((2 / 36.0) * mTop.parameter1, mTemperature[5]);
		mDensity[8].topDirichlet((2 / 36.0) * mTop.parameter1, mDensity[6]);
		mTemperature[8].topDirichlet((2 / 36.0) * mTop.parameter1, mTemperature[6]);
		break;

	default: break;
	}

	switch(mBottom.boundary) {
	case 0:
		mDensity[2].bottomAdiabatic();
		mTemperature[2].bottomAdiabatic();
		mDensity[5].bottomAdiabatic();
		mTemperature[5].bottomAdiabatic();
		mDensity[6].bottomAdiabatic();
		mTemperature[6].bottomAdiabatic();
		break;
	case 1:
		mDensity[2].bottomDirichlet((2 / 9.0) * mBottom.parameter1, mDensity[4]);
		mTemperature[2].bottomDirichlet((2 / 9.0) * mBottom.parameter1, mTemperature[4]);
		mDensity[5].bottomDirichlet((2 / 36.0) * mBottom.parameter1, mDensity[7]);
		mTemperature[5].bottomDirichlet((2 / 36.0) * mBottom.parameter1, mTemperature[7]);
		mDensity[6].bottomDirichlet((2 / 36.0) * mBottom.parameter1, mDensity[8]);
		mTemperature[6].bottomDirichlet((2 / 36.0) * mBottom.parameter1, mTemperature[8]);
		break;

	default: break;
	}

	switch(mLeft.boundary) {
	case 0:
		mDensity[1].leftAdiabatic();
		mTemperature[1].leftAdiabatic();
		mDensity[5].leftAdiabatic();
		mTemperature[5].leftAdiabatic();
		mDensity[8].leftAdiabatic();
		mTemperature[8].leftAdiabatic();
		break;
	case 1:
		mDensity[1].leftDirichlet((2 / 9.0) * mLeft.parameter1, mDensity[3]);
		mTemperature[1].leftDirichlet((2 / 9.0) * mLeft.parameter1, mTemperature[3]);
		mDensity[5].leftDirichlet((2 / 36.0) * mLeft.parameter1, mDensity[7]);
		mTemperature[5].leftDirichlet((2 / 36.0) * mLeft.parameter1, mTemperature[7]);
		mDensity[8].leftDirichlet((2 / 36.0) * mLeft.parameter1, mDensity[6]);
		mTemperature[8].leftDirichlet((2 / 36.0) * mLeft.parameter1, mTemperature[6]);
		break;

	default: break;
	}

	switch(mRight.boundary) {
	case 0:
		mDensity[3].rightAdiabatic();
		mTemperature[3].rightAdiabatic();
		mDensity[6].rightAdiabatic();
		mTemperature[6].rightAdiabatic();
		mDensity[7].rightAdiabatic();
		mTemperature[7].rightAdiabatic();
		break;
	case 1:
		mDensity[3].rightDirichlet((2 / 9.0) * mRight.parameter1, mDensity[1]);
		mTemperature[3].rightDirichlet((2 / 9.0) * mRight.parameter1, mTemperature[1]);
		mDensity[6].rightDirichlet((2 / 36.0) * mRight.parameter1, mDensity[8]);
		mTemperature[6].rightDirichlet((2 / 36.0) * mRight.parameter1, mTemperature[8]);
		mDensity[7].rightDirichlet((2 / 36.0) * mRight.parameter1, mDensity[5]);
		mTemperature[7].rightDirichlet((2 / 36.0) * mRight.parameter1, mTemperature[5]);
		break;

	default: break;
	}
}

void LatticeBoltzmannMethodD2Q9::updateVelocityMatrix()
{
	// TODO: stub
	mVelocityU = Matrix<double>(mWidth, mHeight);
	mVelocityV = Matrix<double>(mWidth, mHeight);
}

void LatticeBoltzmannMethodD2Q9::buildResultingDensityMatrix()
{
	mResultingDensityMatrix = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (4/9) + B * (1/9) + C* (1/9) + D * (1/9) + E * (1/9) + F * (1/36) + G * (1/36) + H * (1/36) + I * (1/36)",
		std::vector<Matrix<double>*>{&mDensity[0],
									 &mDensity[1],
									 &mDensity[2],
									 &mDensity[3],
									 &mDensity[4],
									 &mDensity[5],
									 &mDensity[6],
									 &mDensity[7],
									 &mDensity[8]});
}

void LatticeBoltzmannMethodD2Q9::buildResultingTemperatureMatrix()
{
	mResultingTemperatureMatrix = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (4/9) + B * (1/9) + C* (1/9) + D * (1/9) + E * (1/9) + F * (1/36) + G * (1/36) + H * (1/36) + I * (1/36)",
		std::vector<Matrix<double>*>{&mTemperature[0],
									 &mTemperature[1],
									 &mTemperature[2],
									 &mTemperature[3],
									 &mTemperature[4],
									 &mTemperature[5],
									 &mTemperature[6],
									 &mTemperature[7],
									 &mTemperature[8]});
}