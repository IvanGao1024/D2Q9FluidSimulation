#ifndef LATTICE_BOLTZMANN_METHOD_D2Q9
#define LATTICE_BOLTZMANN_METHOD_D2Q9

#include "Matrix.hpp"
#include <array>
#include <memory>

/**
 * @brief D2Q9 Meta Data Description
 *
 * - f_0: corresponding to speed c(0,0)
 * - f_1: corresponding to speed c(1,0)
 * - f_2: corresponding to speed c(0,1)
 * - f_3: corresponding to speed c(-1,0)
 * - f_4: corresponding to speed c(0,-1)
 * - f_5: corresponding to speed c(1,1)
 * - f_6: corresponding to speed c(-1,1)
 * - f_7: corresponding to speed c(-1,-1)
 * - f_8: corresponding to speed c(1,-1)
 *
 * Constants:
 * - DT: 1
 * - DX: 1
 * - DY: 1
 * - D: 3 (2 + 1)
 * - K_SIZE: 9
 * - C_SPEED_SQUARED_INVERSE: 3 (Assuming dx = dy = dt)
 * - WEIGHT: {4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36}
 */
class LatticeBoltzmannMethodD2Q9
{
	static inline constexpr unsigned int MATRIX_SIZE = 9;  // the number of direction

public:
	enum BoundaryType { NO, ADIABATIC, CONSTANT, BOUNCEBACK, OPEN };
	struct Boundary {
		BoundaryType boundary;
		int          parameter1;
		int          parameter2;

		Boundary(): boundary(BoundaryType::NO), parameter1(0), parameter2(0)
		{
		}
		Boundary(BoundaryType type): boundary(type), parameter1(0), parameter2(0)
		{
		}
		Boundary(BoundaryType type, int param1): boundary(type), parameter1(param1), parameter2(0)
		{
		}
		Boundary(BoundaryType type, int param1, int param2): boundary(type), parameter1(param1), parameter2(param2)
		{
		}
	};

	// 	struct Entity {
	// 		Boundary mTop;
	// 		Boundary mBottom;
	// 		Boundary mLeft;
	// 		Boundary mRight;

	// 		Entity(Boundary bound = Boundary())
	// 		{
	// 			mTop    = bound;
	// 			mBottom = bound;
	// 			mLeft   = bound;
	// 			mRight  = bound;
	// 		}

	// 		Entity(Boundary top, Boundary bottom, Boundary left, Boundary right)
	// 		{
	// 			mTop    = top;
	// 			mBottom = bottom;
	// 			mLeft   = left;
	// 			mRight  = right;
	// 		}
	// 	};

public:
	unsigned int mHeight;
	unsigned int mWidth;

private:
	unsigned int mLength;
	Boundary     mTop;
	Boundary     mBottom;
	Boundary     mLeft;
	Boundary     mRight;

private:  // Internal data
	bool                      mKinematicViscosityArrayRevised;
	bool                      mDiffusionCoefficientArrayRevised;
	std::vector<unsigned int> mKinematicViscosityArray;
	std::vector<unsigned int> mDiffusionCoefficientArray;
	Matrix<unsigned int>      mDensity[MATRIX_SIZE];
	Matrix<unsigned int>      mTemperature[MATRIX_SIZE];

private:                                 // Derived data
	std::vector<unsigned int> mOmega_m;  // density
	std::vector<unsigned int> mOmega_s;  // temperature
	std::vector<unsigned int> mVelocityU;
	std::vector<unsigned int> mVelocityV;
	std::vector<unsigned int> mResultU2;   // u^2
	std::vector<unsigned int> mResultV2;   // v^2
	std::vector<unsigned int> mResultUV2;  // u^2 + v^2

public:  // Pre allocate memory for output
	std::vector<unsigned int> mResultingDensityArray;
	std::vector<unsigned int> mResultingTemperatureArray;

public:
	// 	// Blocks
	// 	std::vector<Entity> mEntities;
	// Parameter
	// 	// Source Matrix
	// 	CartesianMatrix<unsigned int> mDensitySourceMatrix;
	// 	CartesianMatrix<unsigned int> mTemperatureSourceMatrix;
	// std::vector<unsigned int> mVelocityUSourceArray;
	// std::vector<unsigned int> mVelocityVSourceArray;

public:
	LatticeBoltzmannMethodD2Q9(unsigned int              height,
							   unsigned int              width,
							   Boundary                  top,
							   Boundary                  bottom,
							   Boundary                  left,
							   Boundary                  right,
							   std::vector<unsigned int> kinematicViscosityArray,
							   std::vector<unsigned int> diffusionCoefficientArray,
							   std::vector<unsigned int> initialDensityArray     = std::vector<unsigned int>(),
							   std::vector<unsigned int> initialTemperatureArray = std::vector<unsigned int>());

	void step(bool saveImage = false);
	void buildResultingDensityMatrix();
	void buildResultingTemperatureMatrix();

private:
	void collision();
	void streaming();

private:  // helper
	void updateVelocityMatrix();
};
#endif  // LATTICE_BOLTZMANN_METHOD_D2Q9