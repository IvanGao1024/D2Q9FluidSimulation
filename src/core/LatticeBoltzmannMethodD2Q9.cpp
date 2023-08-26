#include "LatticeBoltzmannMethodD2Q9.h"

#include <omp.h>
#include <cassert>

#include "HeatMap.hpp"
#include "OpenCLMain.hpp"

LatticeBoltzmannMethodD2Q9::LatticeBoltzmannMethodD2Q9(unsigned int              height,
													   unsigned int              width,
													   Boundary                  top,
													   Boundary                  bottom,
													   Boundary                  left,
													   Boundary                  right,
													   std::vector<unsigned int> kinematicViscosityArray,
													   std::vector<unsigned int> diffusionCoefficientArray,
													   std::vector<unsigned int> initialDensityArray,
													   std::vector<unsigned int> initialTemperatureArray)
{
	mHeight = height + 1;
	mWidth  = width + 1;
	mLength = mHeight * mWidth;
	mTop    = top;
	mBottom = bottom;
	mLeft   = left;
	mRight  = right;
	// mEntities = entities;

	mKinematicViscosityArrayRevised   = true;
	mDiffusionCoefficientArrayRevised = true;
	mKinematicViscosityArray          = kinematicViscosityArray;
	mDiffusionCoefficientArray        = diffusionCoefficientArray;

	if(initialDensityArray.empty()) {
		initialDensityArray.resize(mLength);
	}
	if(initialTemperatureArray.empty()) {
		initialTemperatureArray.resize(mLength);
	}

#pragma omp parallel sections
	{
#pragma omp section
		{mDensity[0] = Matrix<unsigned int>(mWidth, mHeight, initialDensityArray, 44);
}
#pragma omp section
{
	mDensity[1] = Matrix<unsigned int>(mWidth, mHeight, initialDensityArray, 11);
}
#pragma omp section
{
	mDensity[5] = Matrix<unsigned int>(mWidth, mHeight, initialDensityArray, 3);
}
#pragma omp section
{
	mTemperature[0] = Matrix<unsigned int>(mWidth, mHeight, initialTemperatureArray, 44);
}
#pragma omp section
{
	mTemperature[1] = Matrix<unsigned int>(mWidth, mHeight, initialTemperatureArray, 11);
}
#pragma omp section
{
	mTemperature[5] = Matrix<unsigned int>(mWidth, mHeight, initialTemperatureArray, 3);
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

	for (size_t i = 0; i < 9; i++)
	{
		mTemperature[i].print();
		/* code */
	}
	
	if(saveImage) {
		buildResultingTemperatureMatrix();
		Matrix<unsigned int> result2(mHeight, mWidth, mResultingTemperatureArray);
		result2.print();
	}
}

void LatticeBoltzmannMethodD2Q9::collision()
{
	buildResultingDensityMatrix();
	buildResultingTemperatureMatrix();
	if(mKinematicViscosityArrayRevised) {
		mOmega_m = OpenCLMain::instance().evaluateArithmeticFormula(
			"10000 / ((A * 3) + 50)",
			mWidth,
			mHeight,
			std::vector<unsigned int*>{mKinematicViscosityArray.data()});
	}
	if(mDiffusionCoefficientArrayRevised) {
		mOmega_s = OpenCLMain::instance().evaluateArithmeticFormula(
			"10000 / ((A * 3) + 50)",
			mWidth,
			mHeight,
			std::vector<unsigned int*>{mDiffusionCoefficientArray.data()});
	}

	mTemperature[0].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * 1) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[0].getDataData(), mOmega_s.data(), mResultingTemperatureArray.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[0].getShiftIndexPair()}));
	mTemperature[1].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[1].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityU.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[1].getShiftIndexPair()}));
	mTemperature[2].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[2].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityV.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[2].getShiftIndexPair()}));
	mTemperature[3].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[3].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityU.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[3].getShiftIndexPair()}));
	mTemperature[4].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[4].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityV.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[4].getShiftIndexPair()}));
	mTemperature[5].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100 + 300 * E / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[5].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[5].getShiftIndexPair()}));
	mTemperature[6].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100 + 300 * E / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[6].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[6].getShiftIndexPair()}));
	mTemperature[7].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100 - 300 * E / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[7].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[7].getShiftIndexPair()}));
	mTemperature[8].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100 - 300 * E / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[8].getDataData(),
								   mOmega_s.data(),
								   mResultingTemperatureArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[8].getShiftIndexPair()}));
	mResultU2  = OpenCLMain::instance().evaluateArithmeticFormula("A * A / 100",
                                                                 mWidth,
                                                                 mHeight,
                                                                 std::vector<unsigned int*>{mVelocityU.data()});
	mResultV2  = OpenCLMain::instance().evaluateArithmeticFormula("A * A / 100",
                                                                 mWidth,
                                                                 mHeight,
                                                                 std::vector<unsigned int*>{mVelocityV.data()});
	mResultUV2 = OpenCLMain::instance().evaluateArithmeticFormula(
		"A + B",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mResultU2.data(), mResultV2.data()});
	mDensity[0].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 150 * D / 100)) / 1000000",
		mWidth,
		mHeight,

		std::vector<unsigned int*>{mDensity[0].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[0].getShiftIndexPair()}));
	mDensity[1].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100 + 450 * E / 100 - 150 * F / 100)) / 1000000",
		mWidth,
		mHeight,

		std::vector<unsigned int*>{mDensity[1].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mResultU2.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[1].getShiftIndexPair()}));
	mDensity[2].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100 + 450 * E / 100 - 150 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[2].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityV.data(),
								   mResultV2.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[2].getShiftIndexPair()}));
	mDensity[3].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100 + 450 * E / 100 - 150 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[3].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mResultU2.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[3].getShiftIndexPair()}));
	mDensity[4].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100 + 450 * E / 100 - 150 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[4].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityV.data(),
								   mResultV2.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[4].getShiftIndexPair()}));
	mDensity[5].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100 + 300 * E / 100 + 300 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[5].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[5].getShiftIndexPair()}));
	mDensity[6].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100 + 300 * E / 100 + 300 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[6].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[6].getShiftIndexPair()}));
	mDensity[7].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 - 300 * D / 100 - 300 * E / 100 + 300 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[7].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[7].getShiftIndexPair()}));
	mDensity[8].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * (100 - B)) / 100 + (B * 44 * C * (100 + 300 * D / 100 - 300 * E / 100 + 300 * F / 100)) / 1000000",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[8].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mVelocityV.data(),
								   mResultUV2.data()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[8].getShiftIndexPair()}));
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
	case 0: break;
	case 1:
		mDensity[4].topAdiabatic();
		mTemperature[4].topAdiabatic();
		mDensity[7].topAdiabatic();
		mTemperature[7].topAdiabatic();
		mDensity[8].topAdiabatic();
		mTemperature[8].topAdiabatic();
		break;
	case 2:
		mDensity[4].topDirichlet(22 * mTop.parameter1, mDensity[2]);
		mTemperature[4].topDirichlet(22 * mTop.parameter1, mTemperature[2]);
		mDensity[7].topDirichlet(6 * mTop.parameter1, mDensity[5]);
		mTemperature[7].topDirichlet(6 * mTop.parameter1, mTemperature[5]);
		mDensity[8].topDirichlet(6 * mTop.parameter1, mDensity[6]);
		mTemperature[8].topDirichlet(6 * mTop.parameter1, mTemperature[6]);
		break;

	default: break;
	}

	switch(mBottom.boundary) {
	case 0: break;
	case 1:
		mDensity[2].bottomAdiabatic();
		mTemperature[2].bottomAdiabatic();
		mDensity[5].bottomAdiabatic();
		mTemperature[5].bottomAdiabatic();
		mDensity[6].bottomAdiabatic();
		mTemperature[6].bottomAdiabatic();
		break;
	case 2:
		mDensity[2].bottomDirichlet(22 * mBottom.parameter1, mDensity[4]);
		mTemperature[2].bottomDirichlet(22 * mBottom.parameter1, mTemperature[4]);
		mDensity[5].bottomDirichlet(6 * mBottom.parameter1, mDensity[7]);
		mTemperature[5].bottomDirichlet(6 * mBottom.parameter1, mTemperature[7]);
		mDensity[6].bottomDirichlet(6 * mBottom.parameter1, mDensity[8]);
		mTemperature[6].bottomDirichlet(6 * mBottom.parameter1, mTemperature[8]);
		break;

	default: break;
	}

	switch(mLeft.boundary) {
	case 0: break;
	case 1:
		mDensity[1].leftAdiabatic();
		mTemperature[1].leftAdiabatic();
		mDensity[5].leftAdiabatic();
		mTemperature[5].leftAdiabatic();
		mDensity[8].leftAdiabatic();
		mTemperature[8].leftAdiabatic();
		break;
	case 2:
		mDensity[1].leftDirichlet(22 * mLeft.parameter1, mDensity[3]);
		mTemperature[1].leftDirichlet(22 * mLeft.parameter1, mTemperature[3]);
		mDensity[5].leftDirichlet(6 * mLeft.parameter1, mDensity[7]);
		mTemperature[5].leftDirichlet(6 * mLeft.parameter1, mTemperature[7]);
		mDensity[8].leftDirichlet(6 * mLeft.parameter1, mDensity[6]);
		mTemperature[8].leftDirichlet(6 * mLeft.parameter1, mTemperature[6]);
		break;

	default: break;
	}

	switch(mRight.boundary) {
	case 0: break;
	case 1:
		mDensity[3].rightAdiabatic();
		mTemperature[3].rightAdiabatic();
		mDensity[6].rightAdiabatic();
		mTemperature[6].rightAdiabatic();
		mDensity[7].rightAdiabatic();
		mTemperature[7].rightAdiabatic();
		break;
	case 2:
		mDensity[3].rightDirichlet(22 * mRight.parameter1, mDensity[1]);
		mTemperature[3].rightDirichlet(22 * mRight.parameter1, mTemperature[1]);
		mDensity[6].rightDirichlet(6 * mRight.parameter1, mDensity[8]);
		mTemperature[6].rightDirichlet(6 * mRight.parameter1, mTemperature[8]);
		mDensity[7].rightDirichlet(6 * mRight.parameter1, mDensity[5]);
		mTemperature[7].rightDirichlet(6 * mRight.parameter1, mTemperature[5]);
		break;

	default: break;
	}
}

void LatticeBoltzmannMethodD2Q9::updateVelocityMatrix()
{
	// TODO: stub
	// mVelocityU = mVelocityUSourceArray;
	// mVelocityV = mVelocityVSourceArray;
	mVelocityU.resize(mLength);
	mVelocityV.resize(mLength);
}

void LatticeBoltzmannMethodD2Q9::buildResultingDensityMatrix()
{
	mResultingDensityArray = OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * 44 + B * 11 + C* 11 + D * 11 + E * 11 + F * 3 + G * 3 + H * 3 + I * 3) / 100",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mDensity[0].getDataData(),
								   mDensity[1].getDataData(),
								   mDensity[2].getDataData(),
								   mDensity[3].getDataData(),
								   mDensity[4].getDataData(),
								   mDensity[5].getDataData(),
								   mDensity[6].getDataData(),
								   mDensity[7].getDataData(),
								   mDensity[8].getDataData()},
		std::vector<std::pair<unsigned int, unsigned int>>{mDensity[0].getShiftIndexPair(),
														   mDensity[1].getShiftIndexPair(),
														   mDensity[2].getShiftIndexPair(),
														   mDensity[3].getShiftIndexPair(),
														   mDensity[4].getShiftIndexPair(),
														   mDensity[5].getShiftIndexPair(),
														   mDensity[6].getShiftIndexPair(),
														   mDensity[7].getShiftIndexPair(),
														   mDensity[8].getShiftIndexPair()});
}

void LatticeBoltzmannMethodD2Q9::buildResultingTemperatureMatrix()
{
	mResultingTemperatureArray = OpenCLMain::instance().evaluateArithmeticFormula(
		"(A * 44 + B * 11 + C* 11 + D * 11 + E * 11 + F * 3 + G * 3 + H * 3 + I * 3) / 100",
		mWidth,
		mHeight,
		std::vector<unsigned int*>{mTemperature[0].getDataData(),
								   mTemperature[1].getDataData(),
								   mTemperature[2].getDataData(),
								   mTemperature[3].getDataData(),
								   mTemperature[4].getDataData(),
								   mTemperature[5].getDataData(),
								   mTemperature[6].getDataData(),
								   mTemperature[7].getDataData(),
								   mTemperature[8].getDataData()},
		std::vector<std::pair<unsigned int, unsigned int>>{mTemperature[0].getShiftIndexPair(),
														   mTemperature[1].getShiftIndexPair(),
														   mTemperature[2].getShiftIndexPair(),
														   mTemperature[3].getShiftIndexPair(),
														   mTemperature[4].getShiftIndexPair(),
														   mTemperature[5].getShiftIndexPair(),
														   mTemperature[6].getShiftIndexPair(),
														   mTemperature[7].getShiftIndexPair(),
														   mTemperature[8].getShiftIndexPair()});
}