/**
 * @file clientMain.cpp
 * @author ivan gao (ivan.y.gao@outlook.com)
 * @brief
 * @version 0.1
 * @date 2022-09-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef MAIN_CPP
#define MAIN_CPP

#include <QApplication>
#include "LatticeMainWindow.h"

const quint16 HEIGHT = 800;
const quint16 WIDTH  = 800;

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	LatticeMainWindow mLatticeMainWindow(HEIGHT, WIDTH);
	mLatticeMainWindow.show();

	return QApplication::exec();
}
#endif  // MAIN_CPP
