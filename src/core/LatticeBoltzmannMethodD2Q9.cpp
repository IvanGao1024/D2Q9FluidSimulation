#include "LatticeBoltzmannMethodD2Q9.h"

#include <omp.h>
#include <cassert>

#include "HeatMap.hpp"

LatticeBoltzmannMethodD2Q9::LatticeBoltzmannMethodD2Q9(
	unsigned int                             height,
	unsigned int                             width,
	Boundary                                 top,
	Boundary                                 bottom,
	Boundary                                 left,
	Boundary                                 right,
	std::unique_ptr<CartesianMatrix<double>> initialDensityMatrix,
	std::unique_ptr<CartesianMatrix<double>> initialTemperatureMatrix,
	std::unique_ptr<CartesianMatrix<double>> initialKinematicViscosityMatrix,
	std::unique_ptr<CartesianMatrix<double>> initialDiffusionCoefficientMatrix,
	std::unique_ptr<CartesianMatrix<double>> initialSourceDensityMatrix,
	std::unique_ptr<CartesianMatrix<double>> initialSourceTemperatureMatrix,
	std::vector<Entity>                      entities)
{
	mHeight   = height + 2;
	mWidth    = width + 2;
	mTop      = top;
	mBottom   = bottom;
	mLeft     = left;
	mRight    = right;
	mEntities = entities;

	// Initialize Core Data
	if(initialDensityMatrix) {
		assert(initialDensityMatrix->getHeight() == mHeight && initialDensityMatrix->getWidth() == mWidth);
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i       = omp_get_thread_num();
			mDensity[i] = *initialDensityMatrix * WEIGHT[i];
		}
	} else {
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i       = omp_get_thread_num();
			mDensity[i] = CartesianMatrix<double>(mWidth, mHeight);
		}
	}

	if(initialTemperatureMatrix) {
		assert(initialTemperatureMatrix->getHeight() == mHeight && initialTemperatureMatrix->getWidth() == mWidth);
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i           = omp_get_thread_num();
			mTemperature[i] = *initialTemperatureMatrix * WEIGHT[i];
		}
	} else {
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i           = omp_get_thread_num();
			mTemperature[i] = CartesianMatrix<double>(mWidth, mHeight);
		}
	}

	// Initialize Data
#pragma omp parallel sections
	{
#pragma omp section
		{
			mVelocityU = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mVelocityV = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mResult1 = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mResult2 = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mOmega_m = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mOmega_s = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mResultingTemperatureMatrix = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mResultingDensityMatrix = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			if(initialKinematicViscosityMatrix) {
				assert(initialKinematicViscosityMatrix->getHeight() == mHeight &&
					   initialKinematicViscosityMatrix->getWidth() == mWidth);
				mKinematicViscosityMatrix = *initialKinematicViscosityMatrix;
			} else {
				mKinematicViscosityMatrixRevised = true;
				mKinematicViscosityMatrix =
					CartesianMatrix<double>(mWidth, mHeight, 1.5e-5);  // Air at 20 degree celsius
			}
		}

#pragma omp section
		{
			if(initialDiffusionCoefficientMatrix) {
				assert(initialDiffusionCoefficientMatrix->getHeight() == mHeight &&
					   initialDiffusionCoefficientMatrix->getWidth() == mWidth);
				mDiffusionCoefficientMatrix = *initialDiffusionCoefficientMatrix;
			} else {
				mDiffusionCoefficientMatrixRevised = true;
				mDiffusionCoefficientMatrix =
					CartesianMatrix<double>(mWidth, mHeight, 2.0e-5);  // Air at 20 degree celsius
			}
		}
#pragma omp section
		{
			if(initialSourceDensityMatrix) {
				assert(initialSourceDensityMatrix->getHeight() == mHeight &&
					   initialSourceDensityMatrix->getWidth() == mWidth);
				mDensitySourceMatrix = *initialSourceDensityMatrix;
			} else {
				mDensitySourceMatrix = CartesianMatrix<double>(mWidth, mHeight);
			}
		}
#pragma omp section
		{
			if(initialSourceTemperatureMatrix) {
				assert(initialSourceTemperatureMatrix->getHeight() == mHeight &&
					   initialSourceTemperatureMatrix->getWidth() == mWidth);
				mTemperatureSourceMatrix = *initialSourceTemperatureMatrix;
			} else {
				mTemperatureSourceMatrix = CartesianMatrix<double>(mWidth, mHeight);
			}
		}
#pragma omp section
		{
			mVelocityUSourceMatrix = CartesianMatrix<double>(mWidth, mHeight);
		}
#pragma omp section
		{
			mVelocityVSourceMatrix = CartesianMatrix<double>(mWidth, mHeight);
		}
	}

	// After copying the data, we don't need the initial matrices anymore
	initialDensityMatrix.reset();
	initialTemperatureMatrix.reset();
	initialKinematicViscosityMatrix.reset();
	initialDiffusionCoefficientMatrix.reset();
	initialSourceDensityMatrix.reset();
	initialSourceTemperatureMatrix.reset();
}

void LatticeBoltzmannMethodD2Q9::step(bool buildResult, bool saveImage)
{
	updateVelocityMatrix();
	collision();
	streaming();
	if(buildResult) {
		buildResultDensityMatrix();
		buildResultTemperatureMatrix();
	}
	if(saveImage) {
		// TODO: verify heat map works
		HeatMap::createHeatMap(mResultingDensityMatrix, "density");
		HeatMap::createHeatMap(mResultingTemperatureMatrix, "temperature");
	}
}

void LatticeBoltzmannMethodD2Q9::collision()
{
	buildResultDensityMatrix();
	buildResultTemperatureMatrix();

#pragma omp parallel sections
	{
#pragma omp section
		{if(mKinematicViscosityMatrixRevised){mOmega_m =
												  1.0 / (((mKinematicViscosityMatrix * D) / ((DX * DX) / DT)) + 0.5);
}
}

#pragma omp section
{
	if(mDiffusionCoefficientMatrixRevised) {
		mOmega_s = 1.0 / (((mDiffusionCoefficientMatrix * D) / ((DX * DX) / DT)) + 0.5);
	}
}
}

#pragma omp parallel num_threads(MATRIX_SIZE)
{	
	// TODO: formula is wrong, also super slow.
	int i    = omp_get_thread_num();
	mResult1 = (mVelocityU * C[i].first + mVelocityV * C[i].second) * C_SPEED_SQUARED_INVERSE;
	mResult2 = mResult1 + mResult1 * mResult1 * 0.5 -
			   (mVelocityU * mVelocityU + mVelocityU * mVelocityV * 2 + mVelocityV * mVelocityV) *
				   (0.5 * C_SPEED_SQUARED_INVERSE) +
			   1;

	mDensity[i] = mDensity[i] * (mOmega_m - 1.0) + mOmega_m * (mResultingDensityMatrix * mResult2 * WEIGHT[i]);
	mTemperature[i] =
		mTemperature[i] * (mOmega_s - 1.0) + mOmega_s * (mResultingTemperatureMatrix * (mResult1 + 1) * WEIGHT[i]);
}
}

void LatticeBoltzmannMethodD2Q9::streaming()
{
	// TODO
}

void LatticeBoltzmannMethodD2Q9::updateVelocityMatrix()
{
	// TODO: stub
	mVelocityU = mVelocityUSourceMatrix;
	mVelocityV = mVelocityVSourceMatrix;
}

void LatticeBoltzmannMethodD2Q9::buildResultDensityMatrix()
{
	mResultingDensityMatrix.fill(0);
#pragma omp parallel for
	for(int i = 0; i < MATRIX_SIZE; i++) {
		mResultingDensityMatrix = mResultingDensityMatrix + (mDensity[i] * WEIGHT[i]);
	}
}

void LatticeBoltzmannMethodD2Q9::buildResultTemperatureMatrix()
{
	mResultingTemperatureMatrix.fill(0);
#pragma omp parallel for
	for(int i = 0; i < MATRIX_SIZE; i++) {
		mResultingTemperatureMatrix = mResultingTemperatureMatrix + (mTemperature[i] * WEIGHT[i]);
	}
}
