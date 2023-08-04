#ifndef LATTICE_BOLTZMANN_METHOD_D2Q9
#define LATTICE_BOLTZMANN_METHOD_D2Q9

#include "CartesianMatrix.hpp"

#include <omp.h>  // For OpenMP
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
	static inline constexpr unsigned int          MATRIX_SIZE             = 9;  // the number of direction
	static inline constexpr unsigned int          DT                      = 1;
	static inline constexpr unsigned int          DX                      = 1;
	static inline constexpr unsigned int          DY                      = 1;
	static inline constexpr unsigned int          D                       = 2 + 1;
	static inline constexpr unsigned int          K_SIZE                  = 9;
	static inline constexpr unsigned int          C_SPEED_SQUARED_INVERSE = 3;  // Assume dx = dy = dt
	static inline constexpr std::array<double, 9> WEIGHT =
		{4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};
	static inline constexpr std::array<std::pair<int, int>, 9> C = {{std::make_pair(0, 0),
																	 std::make_pair(1, 0),
																	 std::make_pair(0, 1),
																	 std::make_pair(-1, 0),
																	 std::make_pair(0, -1),
																	 std::make_pair(1, 1),
																	 std::make_pair(-1, 1),
																	 std::make_pair(-1, -1),
																	 std::make_pair(1, -1)}};

public:
	struct Velocity {
		double U;  // Horizontal velocity component
		double V;  // Vertical velocity component
	};

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

	struct Entity {
		Boundary mTop;
		Boundary mBottom;
		Boundary mLeft;
		Boundary mRight;

		Entity(Boundary bound)
		{
			mTop    = bound;
			mBottom = bound;
			mLeft   = bound;
			mRight  = bound;
		}

		Entity(Boundary top, Boundary bottom, Boundary left, Boundary right)
		{
			mTop    = top;
			mBottom = bottom;
			mLeft   = left;
			mRight  = right;
		}
	};

private:
	unsigned int mHeight;
	unsigned int mWidth;
	Boundary     mTop;
	Boundary     mBottom;
	Boundary     mLeft;
	Boundary     mRight;
	// Internal data
	CartesianMatrix<double> mDensity[MATRIX_SIZE];
	CartesianMatrix<double> mTemperature[MATRIX_SIZE];
	// Derived data
	CartesianMatrix<Velocity> mVelocity;

public:  // Pre allocate memory for output
	CartesianMatrix<double> mResultingDensityMatrix;
	CartesianMatrix<double> mResultingTemperatureMatrix;

public:
	// Blocks
	std::vector<Entity> mEntities;
	// Parameter
	CartesianMatrix<double> mKinematicViscosityMatrix;
	CartesianMatrix<double> mDiffusionCoefficientMatrix;
	// Source Matrix
	CartesianMatrix<double>   mDensitySourceMatrix;
	CartesianMatrix<double>   mTemperatureSourceMatrix;
	CartesianMatrix<Velocity> mVelocitySourceMatrix;

public:
	LatticeBoltzmannMethodD2Q9(unsigned int                             height,
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
							   std::vector<Entity>                      entities = std::vector<Entity>());

	void step(std::unique_ptr<CartesianMatrix<Velocity>> SourceVelocityMatrix = nullptr);

private:  // helper
	void updateVelocityMatrix();
	void buildResultDensityMatrix();
	void buildResultTemperatureMatrix();
};
#endif  // LATTICE_BOLTZMANN_METHOD_D2Q9