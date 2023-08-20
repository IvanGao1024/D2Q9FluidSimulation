#ifndef LATTICE_BOLTZMANN_METHOD_D2Q9
#define LATTICE_BOLTZMANN_METHOD_D2Q9

#include "CartesianMatrix.hpp"
#include <array>
#include <memory>

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
	static inline constexpr unsigned int MATRIX_SIZE = 9;  // the number of direction
	// 	static inline constexpr unsigned int          DT                      = 1;
	// 	static inline constexpr unsigned int          DX                      = 1;
	// 	static inline constexpr unsigned int          DY                      = 1;
	// 	static inline constexpr unsigned int          D                       = 2 + 1;
	// 	static inline constexpr unsigned int          K_SIZE                  = 9;
	// 	static inline constexpr unsigned int          C_SPEED_SQUARED_INVERSE = 3;  // Assume dx = dy = dt
	// 	static inline constexpr std::array<double, 9> WEIGHT =
	// 		{4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};
	// 	static inline constexpr std::array<std::pair<int, int>, 9> C = {{std::make_pair(0, 0),
	// 																	 std::make_pair(1, 0),
	// 																	 std::make_pair(0, 1),
	// 																	 std::make_pair(-1, 0),
	// 																	 std::make_pair(0, -1),
	// 																	 std::make_pair(1, 1),
	// 																	 std::make_pair(-1, 1),
	// 																	 std::make_pair(-1, -1),
	// 																	 std::make_pair(1, -1)}};

public:
	enum BoundaryType { NO, BOUNCEBACK, OPEN, CONSTANT, ADIABATIC };
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
	bool                          mKinematicViscosityArrayRevised;
	bool                          mDiffusionCoefficientArrayRevised;
	std::vector<unsigned int>     mKinematicViscosityArray;
	std::vector<unsigned int>     mDiffusionCoefficientArray;
	CartesianMatrix<unsigned int> mDensity[MATRIX_SIZE];
	CartesianMatrix<unsigned int> mTemperature[MATRIX_SIZE];

private:  // Derived data
	std::vector<unsigned int> mOmega_m;
	std::vector<unsigned int> mOmega_s;
	std::vector<unsigned int> mVelocityU;
	std::vector<unsigned int> mVelocityV;
	// std::vector<unsigned int> mResult1;
	// std::vector<unsigned int> mResult2;

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
							   Boundary                  top                       = Boundary(),
							   Boundary                  bottom                    = Boundary(),
							   Boundary                  left                      = Boundary(),
							   Boundary                  right                     = Boundary(),
							   std::vector<unsigned int> initialDensityArray       = std::vector<unsigned int>(),
							   std::vector<unsigned int> initialTemperatureArray   = std::vector<unsigned int>(),
							   std::vector<unsigned int> kinematicViscosityArray   = std::vector<unsigned int>(),
							   std::vector<unsigned int> diffusionCoefficientArray = std::vector<unsigned int>()
							   // std::unique_ptr<CartesianMatrix<unsigned int>>
							   // initialKinematicViscosityMatrix,
							   // std::unique_ptr<CartesianMatrix<unsigned int>> initialDiffusionCoefficientMatrix,
							   // std::unique_ptr<CartesianMatrix<unsigned int>> initialSourceDensityMatrix,
							   // std::unique_ptr<CartesianMatrix<unsigned int>> initialSourceTemperatureMatrix,
							   // std::vector<Entity> entities = std::vector<Entity>()
	);

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