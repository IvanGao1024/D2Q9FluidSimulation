#ifndef LATTICE_BOLTZMANN_METHOD
#define LATTICE_BOLTZMANN_METHOD

#include <QObject>
#include <QDebug>
#include "Matrix.hpp"

class LatticeBoltzmannMethod: public QObject
{
	Q_OBJECT

public:
	LatticeBoltzmannMethod(quint16 height, quint16 width, QObject* parent = nullptr);

	static void increment(int& x)
	{
		++x;
	}
};
#endif  // LATTICE_BOLTZMANN_METHOD