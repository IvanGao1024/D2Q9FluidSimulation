#ifndef LATTICE_BOLTZMANN_METHOD_D2Q9
#define LATTICE_BOLTZMANN_METHOD_D2Q9

#include "CartesianMatrix.hpp"

class LatticeBoltzmannMethodD2Q9
{
	// D2Q9 Meta data
	// - f_0 corresonding to speed c(0,0)
	// - f_1 corresonding to speed c(1,0)
	// - f_2 corresonding to speed c(0,1)
	// - f_3 corresonding to speed c(-1,0)
	// - f_4 corresonding to speed c(0,-1)
	// - f_5 corresonding to speed c(1,1)
	// - f_6 corresonding to speed c(-1,1)
	// - f_7 corresonding to speed c(-1,-1)
	// - f_8 corresonding to speed c(1,-1)
private:
	static inline constexpr unsigned int         MATRIX_SIZE             = 9;  // the number of direction
	static inline constexpr unsigned int         DT                      = 1;
	static inline constexpr unsigned int         DX                      = 1;
	static inline constexpr unsigned int         DY                      = 1;
	static inline constexpr unsigned int         D                       = 2 + 1;
	static inline constexpr unsigned int         K_SIZE                  = 9;
	static inline constexpr unsigned int         C_SPEED_SQUARED_INVERSE = 3;  // Assume dx = dy = dt
	static inline constexpr std::array<float, 9> WEIGHT =
		{1.0F / 9, 1.0F / 9, 1.0F / 9, 1.0F / 9, 1.0F / 36, 1.0F / 36, 1.0F / 36, 1.0F / 36, 4.0F / 9};
	static inline constexpr std::array<std::pair<int, int>, 9> C = {{std::make_pair(0, 0),
                                                                     std::make_pair(1, 0),
																	 std::make_pair(0, 1),
																	 std::make_pair(-1, 0),
																	 std::make_pair(0, -1),
																	 std::make_pair(1, 1),
																	 std::make_pair(-1, 1),
																	 std::make_pair(-1, -1),
																	 std::make_pair(1, -1)}};
	float mOmega_m;
	float mOmega_s;

private:
	CartesianMatrix<float>  mDensity[MATRIX_SIZE];
	CartesianMatrix<float>  mTemperature[MATRIX_SIZE];
	CartesianMatrix<std::pair<float, float>>  mVelocity;
	CartesianMatrix<float> *pDensityMatrix;
	CartesianMatrix<float> *pTemperatureMatrix;
	CartesianMatrix<float> *pVelocityMatrix;

public:
	LatticeBoltzmannMethodD2Q9(unsigned int height, unsigned int width,
		   CartesianMatrix<float> *DensityMatrix,
		   CartesianMatrix<float> *TemperatureMatrix,
		   CartesianMatrix<float> *VelocityMatrix,
		   float                   kinematicViscosity,
		   float                   diffusionCoefficient);
};
#endif  // LATTICE_BOLTZMANN_METHOD_D2Q9