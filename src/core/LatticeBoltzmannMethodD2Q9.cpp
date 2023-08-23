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
													   std::vector<unsigned int> initialDensityArray,
													   std::vector<unsigned int> initialTemperatureArray,
													   std::vector<unsigned int> kinematicViscosityArray,
													   std::vector<unsigned int> diffusionCoefficientArray)
{
	mHeight = height;
	mWidth  = width;
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
	if(saveImage) {
		// TODO: verify heat map works
		// HeatMap::createHeatMap(mResultingDensityMatrix, "density");
		// HeatMap::createHeatMap(mResultingTemperatureMatrix, "temperature");
	}
}

void LatticeBoltzmannMethodD2Q9::collision()
{
	buildResultingDensityMatrix();
	buildResultingTemperatureMatrix();
	if(mKinematicViscosityArrayRevised) {
		mOmega_m = OpenCLMain::instance().evaluateArithmeticFormula(
			"1 / ((A * 3) + 50)",
			mWidth,
			mHeight,
			mLength,
			std::vector<unsigned int*>{mKinematicViscosityArray.data()});
	}
	if(mDiffusionCoefficientArrayRevised) {
		mOmega_s = OpenCLMain::instance().evaluateArithmeticFormula(
			"1 / ((A * 3) + 50)",
			mWidth,
			mHeight,
			mLength,
			std::vector<unsigned int*>{mDiffusionCoefficientArray.data()});
	}

	mTemperature[0].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * 1",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mTemperature[0].getDataData(), mOmega_s.data(), mResultingTemperatureArray.data()}));
	mTemperature[1].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D)",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[1].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data()}));
	mTemperature[2].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D)",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[2].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityV.data()}));
	mTemperature[3].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D)",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[3].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data()}));
	mTemperature[4].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D)",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[4].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityV.data()}));
	mTemperature[5].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D + 3 * E))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[5].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()}));
	mTemperature[6].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D + 3 * E))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[6].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()}));
	mTemperature[7].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D - 3 * E))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[7].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()}));
	mTemperature[8].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D - 3 * E))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mTemperature[8].getDataData(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()}));
	mResultU2  = OpenCLMain::instance().evaluateArithmeticFormula("A * A",
                                                                 mWidth,
                                                                 mHeight,
                                                                 mLength,
                                                                 std::vector<unsigned int*>{mVelocityU.data()});
	mResultV2  = OpenCLMain::instance().evaluateArithmeticFormula("A * A",
                                                                 mWidth,
                                                                 mHeight,
                                                                 mLength,
                                                                 std::vector<unsigned int*>{mVelocityV.data()});
	mResultUV2 = OpenCLMain::instance().evaluateArithmeticFormula(
		"A + B",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mResultU2.data(), mResultV2.data()});
	mDensity[0].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 15 * D))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mDensity[0].getDataData(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mResultUV2.data()}));
	mDensity[1].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 + 3 * D + 45 * E - 15 * F))",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mDensity[1].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mResultU2.data(),
								   mResultUV2.data()}));
	mDensity[2].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 + 3 * D + 45 * E - 15 * F))",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mDensity[2].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityV.data(),
								   mResultV2.data(),
								   mResultUV2.data()}));
	mDensity[3].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 - 3 * D + 45 * E - 15 * F))",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mDensity[3].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mResultU2.data(),
								   mResultUV2.data()}));
	mDensity[4].resetData(OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 - 3 * D + 45 * E - 15 * F))",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mDensity[4].getDataData(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityV.data(),
								   mResultV2.data(),
								   mResultUV2.data()}));
	mDensity[5].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D + 3 * E + 3 * F))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mDensity[5].getDataData(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()}));
	mDensity[6].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D + 3 * E + 3 * F))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mDensity[6].getDataData(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()}));
	mDensity[7].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D - 3 * E + 3 * F))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mDensity[7].getDataData(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()}));
	mDensity[8].resetData(
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D - 3 * E + 3 * F))",
														 mWidth,
														 mHeight,
														 mLength,
														 std::vector<unsigned int*>{mDensity[8].getDataData(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()}));
}

void LatticeBoltzmannMethodD2Q9::streaming()
{
	// TODO
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
		"A * 44 + B * 11 + C* 11 + D * 11 + E * 11 + F * 3 + G * 3 + H * 3 + I * 3",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mDensity[0].getDataData(),
								   mDensity[1].getDataData(),
								   mDensity[2].getDataData(),
								   mDensity[3].getDataData(),
								   mDensity[4].getDataData(),
								   mDensity[5].getDataData(),
								   mDensity[6].getDataData(),
								   mDensity[7].getDataData(),
								   mDensity[8].getDataData()});
}

void LatticeBoltzmannMethodD2Q9::buildResultingTemperatureMatrix()
{
	mResultingTemperatureArray = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * 44 + B * 11 + C* 11 + D * 11 + E * 11 + F * 3 + G * 3 + H * 3 + I * 3",
		mWidth,
		mHeight,
		mLength,
		std::vector<unsigned int*>{mTemperature[0].getDataData(),
								   mTemperature[1].getDataData(),
								   mTemperature[2].getDataData(),
								   mTemperature[3].getDataData(),
								   mTemperature[4].getDataData(),
								   mTemperature[5].getDataData(),
								   mTemperature[6].getDataData(),
								   mTemperature[7].getDataData(),
								   mTemperature[8].getDataData()});
}