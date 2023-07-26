#include "LatticeMainWindow.h"

#include <QLabel>
LatticeMainWindow::LatticeMainWindow(quint16 height, quint16 width, QWidget* parent):
	QMainWindow(parent),
	mHeight(height),
	mWidth(width)
{
	resize(mHeight, mWidth);
	D2Q9(10000, 10000);
}