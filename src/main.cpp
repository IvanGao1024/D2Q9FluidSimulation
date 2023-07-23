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

#include "LatticeApplication.h"

int main(int argc, char *argv[])
{
	LatticeApplication LatticeApplication(argc, argv);
	return LatticeApplication.exec();
}
#endif // MAIN_CPP
