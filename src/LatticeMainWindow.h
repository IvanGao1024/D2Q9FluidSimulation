#ifndef LATTICE_MAINWINDOW
#define LATTICE_MAINWINDOW

#include <QMainWindow>
#include "core/LatticeBoltzmannMethod/D2Q9.h"

class LatticeMainWindow: public QMainWindow
{
	Q_OBJECT

private:
	quint16 mHeight;
	quint16 mWidth;

public:
	LatticeMainWindow(quint16 height, quint16 width, QWidget* parent = nullptr);
};
#endif  // LATTICE_MAINWINDOW