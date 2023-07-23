#ifndef LATTICE_APPLICATION
#define LATTICE_APPLICATION

#include <QApplication>

class LatticeApplication: public QApplication
{
	Q_OBJECT

public:
	LatticeApplication(int argc, char *argv[]);
};
#endif //LATTICE_APPLICATION