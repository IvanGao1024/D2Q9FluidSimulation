#include "LatticeBoltzmannMethodD2Q9.h"

#include <omp.h>

LatticeBoltzmannMethodD2Q9::LatticeBoltzmannMethodD2Q9(unsigned int height,
		   unsigned int width,
		   CartesianMatrix<float> *DensityMatrix,
		   CartesianMatrix<float> *TemperatureMatrix,
		   CartesianMatrix<float> *VelocityMatrix,
		   float                   kinematicViscosity,
		   float                   diffusionCoefficient)
{
    pDensityMatrix = DensityMatrix;
    pTemperatureMatrix = TemperatureMatrix;
    pVelocityMatrix = VelocityMatrix;

	mOmega_m = 1.0 / (((D * kinematicViscosity) / ((DX * DX) / DT)) + 0.5);
	mOmega_s = 1.0 / (((D * diffusionCoefficient) / ((DX * DX) / DT)) + 0.5);

    #pragma omp parallel num_threads(MATRIX_SIZE)
    {
        int i = omp_get_thread_num();
        mDensity[i] = CartesianMatrix<float>(width + 1, height + 1, 0.0F);
    }

    #pragma omp parallel num_threads(MATRIX_SIZE)
    {
        int i = omp_get_thread_num();
        mTemperature[i] = CartesianMatrix<float>(width + 1, height + 1, 0.0F);
    }

    mVelocity = CartesianMatrix<std::pair<float, float>>(width + 1, height + 1, {0.0F, 0.0F});

    // @threads for k in 1:klim
    //     f[:, :, k] = weight[k] .* InitialConditions[:, :, 1]
    //     g[:, :, k] = weight[k] .* InitialConditions[:, :, 2]
    // end

    // # Initiate helper data structure
    // temp = fill(0.0, (N, M))
    // feq = fill(0.0, (N, M))
    // f_updated = similar(f)
    // result = zeros(N, M)
    
}