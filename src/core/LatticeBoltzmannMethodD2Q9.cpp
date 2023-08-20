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
// LatticeBoltzmannMethodD2Q9::LatticeBoltzmannMethodD2Q9(
// 	unsigned int                                   height,
// 	unsigned int                                   width,
// 	Boundary                                       top,
// 	Boundary                                       bottom,
// 	Boundary                                       left,
// 	Boundary                                       right,
// 	std::unique_ptr<CartesianMatrix<unsigned int>> initialDensityMatrix,
// 	std::unique_ptr<CartesianMatrix<unsigned int>> initialTemperatureMatrix,
// 	std::unique_ptr<CartesianMatrix<unsigned int>> initialKinematicViscosityMatrix,
// 	std::unique_ptr<CartesianMatrix<unsigned int>> initialDiffusionCoefficientMatrix,
// 	std::unique_ptr<CartesianMatrix<unsigned int>> initialSourceDensityMatrix,
// 	std::unique_ptr<CartesianMatrix<unsigned int>> initialSourceTemperatureMatrix,
// 	std::vector<Entity>                            entities)
// {
// 	mHeight   = height + 2;
// 	mWidth    = width + 2;
// 	mLength   = mHeight * mWidth;
// 	mTop      = top;
// 	mBottom   = bottom;
// 	mLeft     = left;
// 	mRight    = right;
// 	mEntities = entities;

// 	mDensity[0] =
// 		CartesianMatrix<unsigned int>(mWidth, mHeight,
// 			OpenCLMain::instance().evaluateArithmeticFormula("A * 44",
// 															 mLength,
// 															 std::vector<unsigned int*>{
// 																initialDensityMatrix->data.data()});
// 	mDensity[1] =
// 		CartesianMatrix<unsigned int>(mWidth, mHeight,
// 			OpenCLMain::instance().evaluateArithmeticFormula("A * 11",
// 															 mLength,
// 															 std::vector<unsigned int*>{
// 																initialDensityMatrix->data.data()});

// 	mDensity[2] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[1].data);
// 	mDensity[3] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[1].data);
// 	mDensity[4] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[1].data);
// 	mDensity[5] =
// 		CartesianMatrix<unsigned int>(mWidth, mHeight,
// 			OpenCLMain::instance().evaluateArithmeticFormula("A * 3",
// 															 mLength,
// 															 std::vector<unsigned int*>{
// 																initialDensityMatrix->data.data()});
// 	mDensity[6] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[5].data);
// 	mDensity[7] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[5].data);
// 	mDensity[8] = CartesianMatrix<unsigned int>(mWidth, mHeight, mDensity[5].data);

// 	if(initialTemperatureMatrix) {
// 		assert(initialTemperatureMatrix->getHeight() == mHeight && initialTemperatureMatrix->getWidth() == mWidth);
// #pragma omp parallel num_threads(MATRIX_SIZE)
// 		{
// 			int i           = omp_get_thread_num();
// 			mTemperature[i] = *initialTemperatureMatrix * WEIGHT[i];
// 		}
// 	} else {
// #pragma omp parallel num_threads(MATRIX_SIZE)
// 		{
// 			int i           = omp_get_thread_num();
// 			mTemperature[i] = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// 	}

// 	// Initialize Data
// #pragma omp parallel sections
// 	{
// #pragma omp section
// 		{
// 			mVelocityU = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mVelocityV = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mResult1 = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mResult2 = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mOmega_m = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mOmega_s = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mResultingTemperatureMatrix = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mResultingDensityMatrix = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			if(initialKinematicViscosityMatrix) {
// 				assert(initialKinematicViscosityMatrix->getHeight() == mHeight &&
// 					   initialKinematicViscosityMatrix->getWidth() == mWidth);
// 				mKinematicViscosityMatrix = *initialKinematicViscosityMatrix;
// 			} else {
// 				mKinematicViscosityMatrixRevised = true;
// 				mKinematicViscosityMatrix =
// 					CartesianMatrix<unsigned int>(mWidth, mHeight, 1.5e-5);  // Air at 20 degree celsius
// 			}
// 		}

// #pragma omp section
// 		{
// 			if(initialDiffusionCoefficientMatrix) {
// 				assert(initialDiffusionCoefficientMatrix->getHeight() == mHeight &&
// 					   initialDiffusionCoefficientMatrix->getWidth() == mWidth);
// 				mDiffusionCoefficientMatrix = *initialDiffusionCoefficientMatrix;
// 			} else {
// 				mDiffusionCoefficientMatrixRevised = true;
// 				mDiffusionCoefficientMatrix =
// 					CartesianMatrix<unsigned int>(mWidth, mHeight, 2.0e-5);  // Air at 20 degree celsius
// 			}
// 		}
// #pragma omp section
// 		{
// 			if(initialSourceDensityMatrix) {
// 				assert(initialSourceDensityMatrix->getHeight() == mHeight &&
// 					   initialSourceDensityMatrix->getWidth() == mWidth);
// 				mDensitySourceMatrix = *initialSourceDensityMatrix;
// 			} else {
// 				mDensitySourceMatrix = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 			}
// 		}
// #pragma omp section
// 		{
// 			if(initialSourceTemperatureMatrix) {
// 				assert(initialSourceTemperatureMatrix->getHeight() == mHeight &&
// 					   initialSourceTemperatureMatrix->getWidth() == mWidth);
// 				mTemperatureSourceMatrix = *initialSourceTemperatureMatrix;
// 			} else {
// 				mTemperatureSourceMatrix = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 			}
// 		}
// #pragma omp section
// 		{
// 			mVelocityUSourceMatrix = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// #pragma omp section
// 		{
// 			mVelocityVSourceMatrix = CartesianMatrix<unsigned int>(mWidth, mHeight);
// 		}
// 	}

// 	// After copying the data, we don't need the initial matrices anymore
// 	initialDensityMatrix.reset();
// 	initialTemperatureMatrix.reset();
// 	initialKinematicViscosityMatrix.reset();
// 	initialDiffusionCoefficientMatrix.reset();
// 	initialSourceDensityMatrix.reset();
// 	initialSourceTemperatureMatrix.reset();
// }

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
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data()});
	mTemperature[1].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 + 3 * D)",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityU.data()});
	mTemperature[2].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 + 3 * D)",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityV.data()});
	mTemperature[3].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 - 3 * D)",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityU.data()});
	mTemperature[4].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 - 3 * D)",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityV.data()});
	mTemperature[5].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 + 3 * D + 3 * E))",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityU.data(),
									   mVelocityV.data()});
	mTemperature[6].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 - 3 * D + 3 * E))",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityU.data(),
									   mVelocityV.data()});
	mTemperature[7].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 - 3 * D - 3 * E))",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityU.data(),
									   mVelocityV.data()});
	mTemperature[8].data = OpenCLMain::instance().evaluateArithmeticFormula(
	        "A * (100 - B) + B * 44 * C * (100 + 3 * D - 3 * E))",
	        mLength,
	        std::vector<unsigned int*>{mTemperature[0].data.data(),
									   mOmega_s.data(),
									   mResultingTemperatureArray.data(),
									   mVelocityU.data(),
									   mVelocityV.data()});

	// mResult1.data = OpenCLMain::instance().evaluateArithmeticFormula(
	//             "(A *  + B * b) * 3",
	//             8192*8192,
	//             std::vector<unsigned int*>{mDiffusionCoefficientMatrix.data.data()},
	//             std::vector<unsigned int*>{C[i].first, C[i].second});

	// 	mResult1 = (mVelocityU * C[i].first + mVelocityV * C[i].second) * C_SPEED_SQUARED_INVERSE;

	// 		mTemperature[i] * (mOmega_s - 1.0) + mOmega_s * (mResultingTemperatureMatrix * (mResult1 + 1) * WEIGHT[i]);
	// mResult2 = mResult1 + mResult1 * mResult1 * 0.5 -
	// 			   (mVelocityU * mVelocityU + mVelocityU * mVelocityV * 2 + mVelocityV * mVelocityV) *
	// 				   (0.5 * C_SPEED_SQUARED_INVERSE) +
	// 			   1;

	// 	mDensity[i] = mDensity[i] * (mOmega_m - 1.0) + mOmega_m * (mResultingDensityMatrix * mResult2 * WEIGHT[i]);
	// 	mTemperature[i] =
	// 		mTemperature[i] * (mOmega_s - 1.0) + mOmega_s * (mResultingTemperatureMatrix * (mResult1 + 1) * WEIGHT[i]);
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