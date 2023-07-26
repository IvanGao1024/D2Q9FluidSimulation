#ifndef LATTICE_BOLTZMANN_METHOD
#define LATTICE_BOLTZMANN_METHOD

#include <QObject>
#include <QDebug>

#include "../CartesianMatrix.hpp"

class D2Q9: public QObject
{
	Q_OBJECT

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
	static inline constexpr unsigned int dt = 1;
	static inline constexpr unsigned int dx = 1;
	static inline constexpr unsigned int dy = 1;
	static inline constexpr unsigned int D = 2+1;

    static inline constexpr std::array<float, 9> weight = {1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/36, 1.0f/36, 1.0f/36, 1.0f/36, 4.0f/9};
    static inline constexpr std::array<std::array<int, 2>, 9> c = {{{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}, {0, 0}}};

    // klim = length(weight)
    // omega = 1.0 / (((D * alpha) / ((dx*dx) / dt)) + 0.5)
    // c_k = [dx / dt, dy / dt]
    // c_s_squared_inverse = 3 # Assume dx = dy = dt
    
private:
	CartesianMatrix<float> mData[MATRIX_SIZE];

public:
	D2Q9(quint16 height, quint16 width, QObject* parent = nullptr);
};
#endif  // LATTICE_BOLTZMANN_METHOD