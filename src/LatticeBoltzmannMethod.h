#ifndef LATTICE_BOLTZMANN_METHOD
#define LATTICE_BOLTZMANN_METHOD

#include <QObject>

class LatticeBoltzmannMethod: public QObject
{
	Q_OBJECT

public:
	LatticeBoltzmannMethod(QObject *parent = nullptr);
};
#endif //LATTICE_BOLTZMANN_METHOD