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

	mDensity[0]     = CartesianMatrix<unsigned int>(mWidth, mHeight, initialDensityArray, 44);
	mDensity[1]     = CartesianMatrix<unsigned int>(mWidth, mHeight, initialDensityArray, 11);
	mDensity[5]     = CartesianMatrix<unsigned int>(mWidth, mHeight, initialDensityArray, 3);
	mTemperature[0] = CartesianMatrix<unsigned int>(mWidth, mHeight, initialTemperatureArray, 44);
	mTemperature[1] = CartesianMatrix<unsigned int>(mWidth, mHeight, initialTemperatureArray, 11);
	mTemperature[5] = CartesianMatrix<unsigned int>(mWidth, mHeight, initialTemperatureArray, 3);

#pragma omp parallel sections
	{
#pragma omp section
		{
			mDensity[2] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[1].data);
		}
#pragma omp section
		{
			mDensity[3] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[1].data);
		}
#pragma omp section
		{
			mDensity[4] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[1].data);
		}
#pragma omp section
		{
			mDensity[6] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[5].data);
		}
#pragma omp section
		{
			mDensity[7] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[5].data);
		}
#pragma omp section
		{
			mDensity[8] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[5].data);
		}
#pragma omp section
		{
			mTemperature[2] = CartesianMatrix<unsigned int>(mWidth, mHeight, mTemperature[1].data);
		}
#pragma omp section
		{
			mTemperature[3] = CartesianMatrix<unsigned int>(mWidth, mHeight, mTemperature[1].data);
		}
#pragma omp section
		{
			mTemperature[4] = CartesianMatrix<unsigned int>(mWidth, mHeight, mTemperature[1].data);
		}
#pragma omp section
		{
			mTemperature[6] = CartesianMatrix<unsigned int>(mWidth, mHeight, mTemperature[5].data);
		}
#pragma omp section
		{
			mTemperature[7] = CartesianMatrix<unsigned int>(mWidth, mHeight, mTemperature[5].data);
		}
#pragma omp section
		{
			mTemperature[8] = CartesianMatrix<unsigned int>(mWidth, mHeight, mTemperature[5].data);
		}
	}
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
			mLength,
			std::vector<unsigned int*>{mKinematicViscosityArray.data()});
	}
	if(mDiffusionCoefficientArrayRevised) {
		mOmega_s = OpenCLMain::instance().evaluateArithmeticFormula(
			"1 / ((A * 3) + 50)",
			mLength,
			std::vector<unsigned int*>{mDiffusionCoefficientArray.data()});
	}

	mTemperature[0].data = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * 1",
		mLength,
		std::vector<unsigned int*>{mTemperature[0].data.data(), mOmega_s.data(), mResultingTemperatureArray.data()});
	mTemperature[1].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D)",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[1].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data()});
	mTemperature[2].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D)",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[2].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityV.data()});
	mTemperature[3].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D)",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[3].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data()});
	mTemperature[4].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D)",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[4].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityV.data()});
	mTemperature[5].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D + 3 * E))",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[5].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()});
	mTemperature[6].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D + 3 * E))",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[6].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()});
	mTemperature[7].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D - 3 * E))",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[7].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()});
	mTemperature[8].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D - 3 * E))",
														 mLength,
														 std::vector<unsigned int*>{mTemperature[8].data.data(),
																					mOmega_s.data(),
																					mResultingTemperatureArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data()});
	mResultU2  = OpenCLMain::instance().evaluateArithmeticFormula("A * A",
                                                                 mLength,
                                                                 std::vector<unsigned int*>{mVelocityU.data()});
	mResultV2  = OpenCLMain::instance().evaluateArithmeticFormula("A * A",
                                                                 mLength,
                                                                 std::vector<unsigned int*>{mVelocityV.data()});
	mResultUV2 = OpenCLMain::instance().evaluateArithmeticFormula(
		"A + B",
		mLength,
		std::vector<unsigned int*>{mResultU2.data(), mResultV2.data()});
	mDensity[0].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 15 * D))",
														 mLength,
														 std::vector<unsigned int*>{mDensity[0].data.data(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mResultUV2.data()});
	mDensity[1].data = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 + 3 * D + 45 * E - 15 * F))",
		mLength,
		std::vector<unsigned int*>{mDensity[1].data.data(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mResultU2.data(),
								   mResultUV2.data()});
	mDensity[2].data = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 + 3 * D + 45 * E - 15 * F))",
		mLength,
		std::vector<unsigned int*>{mDensity[2].data.data(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityV.data(),
								   mResultV2.data(),
								   mResultUV2.data()});
	mDensity[3].data = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 - 3 * D + 45 * E - 15 * F))",
		mLength,
		std::vector<unsigned int*>{mDensity[3].data.data(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityU.data(),
								   mResultU2.data(),
								   mResultUV2.data()});
	mDensity[4].data = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * (100 - B) + B * 44 * C * (100 - 3 * D + 45 * E - 15 * F))",
		mLength,
		std::vector<unsigned int*>{mDensity[4].data.data(),
								   mOmega_m.data(),
								   mResultingDensityArray.data(),
								   mVelocityV.data(),
								   mResultV2.data(),
								   mResultUV2.data()});
	mDensity[5].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D + 3 * E + 3 * F))",
														 mLength,
														 std::vector<unsigned int*>{mDensity[5].data.data(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()});
	mDensity[6].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D + 3 * E + 3 * F))",
														 mLength,
														 std::vector<unsigned int*>{mDensity[6].data.data(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()});
	mDensity[7].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 - 3 * D - 3 * E + 3 * F))",
														 mLength,
														 std::vector<unsigned int*>{mDensity[7].data.data(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()});
	mDensity[8].data =
		OpenCLMain::instance().evaluateArithmeticFormula("A * (100 - B) + B * 44 * C * (100 + 3 * D - 3 * E + 3 * F))",
														 mLength,
														 std::vector<unsigned int*>{mDensity[8].data.data(),
																					mOmega_m.data(),
																					mResultingDensityArray.data(),
																					mVelocityU.data(),
																					mVelocityV.data(),
																					mResultUV2.data()});
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
		mLength,
		std::vector<unsigned int*>{mDensity[0].data.data(),
								   mDensity[1].data.data(),
								   mDensity[2].data.data(),
								   mDensity[3].data.data(),
								   mDensity[4].data.data(),
								   mDensity[5].data.data(),
								   mDensity[6].data.data(),
								   mDensity[7].data.data(),
								   mDensity[8].data.data()});
}

void LatticeBoltzmannMethodD2Q9::buildResultingTemperatureMatrix()
{
	mResultingTemperatureArray = OpenCLMain::instance().evaluateArithmeticFormula(
		"A * 44 + B * 11 + C* 11 + D * 11 + E * 11 + F * 3 + G * 3 + H * 3 + I * 3",
		mLength,
		std::vector<unsigned int*>{mTemperature[0].data.data(),
								   mTemperature[1].data.data(),
								   mTemperature[2].data.data(),
								   mTemperature[3].data.data(),
								   mTemperature[4].data.data(),
								   mTemperature[5].data.data(),
								   mTemperature[6].data.data(),
								   mTemperature[7].data.data(),
								   mTemperature[8].data.data()});
}