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
	mHeight   = height;
	mWidth    = width;
	mTop      = top;
	mBottom   = bottom;
	mLeft     = left;
	mRight    = right;
	mEntities = entities;

	// Initialize Core Data
	if(initialDensityMatrix) {
		assert(initialDensityMatrix->getHeight() == height + 2 && initialDensityMatrix->getWidth() == width + 2);
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i       = omp_get_thread_num();
			mDensity[i] = *initialDensityMatrix * WEIGHT[i];
		}
	} else {
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i       = omp_get_thread_num();
			mDensity[i] = CartesianMatrix<double>(width + 2, height + 2);
		}
	}

	if(initialTemperatureMatrix) {
		assert(initialTemperatureMatrix->getHeight() == height + 2 &&
			   initialTemperatureMatrix->getWidth() == width + 2);
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i           = omp_get_thread_num();
			mTemperature[i] = *initialTemperatureMatrix * WEIGHT[i];
		}
	} else {
#pragma omp parallel num_threads(MATRIX_SIZE)
		{
			int i           = omp_get_thread_num();
			mTemperature[i] = CartesianMatrix<double>(width + 2, height + 2);
		}
	}

	// Initialize Data
#pragma omp parallel sections
	{
#pragma omp section
		{
			mVelocity = CartesianMatrix<Velocity>(width + 2, height + 2);
		}
#pragma omp section
		{
			mResultingTemperatureMatrix = CartesianMatrix<double>(width + 2, height + 2);
		}
#pragma omp section
		{
			mResultingDensityMatrix = CartesianMatrix<double>(width + 2, height + 2);
		}
#pragma omp section
		{
			if(initialKinematicViscosityMatrix) {
				assert(initialKinematicViscosityMatrix->getHeight() == height + 2 &&
					   initialKinematicViscosityMatrix->getWidth() == width + 2);
				mKinematicViscosityMatrix = *initialKinematicViscosityMatrix;
			} else {
				mKinematicViscosityMatrix =
					CartesianMatrix<double>(width + 2, height + 2, 1.5e-5);  // Air at 20 degree celsius
			}
		}

#pragma omp section
		{
			if(initialDiffusionCoefficientMatrix) {
				assert(initialDiffusionCoefficientMatrix->getHeight() == height + 2 &&
					   initialDiffusionCoefficientMatrix->getWidth() == width + 2);
				mDiffusionCoefficientMatrix = *initialDiffusionCoefficientMatrix;
			} else {
				mDiffusionCoefficientMatrix =
					CartesianMatrix<double>(width + 2, height + 2, 2.0e-5);  // Air at 20 degree celsius
			}
		}
#pragma omp section
		{
			if(initialSourceDensityMatrix) {
				assert(initialSourceDensityMatrix->getHeight() == height + 2 &&
					   initialSourceDensityMatrix->getWidth() == width + 2);
				mDensitySourceMatrix = *initialSourceDensityMatrix;
			} else {
				mDensitySourceMatrix = CartesianMatrix<double>(width + 2, height + 2);
			}
		}
#pragma omp section
		{
			if(initialSourceTemperatureMatrix) {
				assert(initialSourceTemperatureMatrix->getHeight() == height + 2 &&
					   initialSourceTemperatureMatrix->getWidth() == width + 2);
				mTemperatureSourceMatrix = *initialSourceTemperatureMatrix;
			} else {
				mTemperatureSourceMatrix = CartesianMatrix<double>(width + 2, height + 2);
			}
		}
#pragma omp section
		{
			mVelocitySourceMatrix = CartesianMatrix<Velocity>(width + 2, height + 2);
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

void LatticeBoltzmannMethodD2Q9::step(std::unique_ptr<CartesianMatrix<Velocity>> SourceVelocityMatrix)
{
	updateVelocityMatrix();

	// mOmega_m = 1.0 / (((D * kinematicViscosity) / ((DX * DX) / DT)) + 0.5);
	// mOmega_s = 1.0 / (((D * diffusionCoefficient) / ((DX * DX) / DT)) + 0.5);
	buildResultDensityMatrix();
	buildResultTemperatureMatrix();
	HeatMap::createHeatMap(mResultingDensityMatrix, "density");
	HeatMap::createHeatMap(mResultingTemperatureMatrix, "temperature");
}

void LatticeBoltzmannMethodD2Q9::updateVelocityMatrix()
{
	// TODO: stub
	mVelocity = CartesianMatrix<Velocity>(mWidth + 2, mHeight + 2);
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
