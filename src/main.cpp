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

#include "core/LatticeBoltzmannMethodD2Q9.h"

int main(int argc, char *argv[])
{
	Matrix<double> m1(8, 8, 0.25);
	LatticeBoltzmannMethodD2Q9 lbm (7, 7,
		LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
		LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::ADIABATIC),
		LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 0),
		LatticeBoltzmannMethodD2Q9::Boundary(LatticeBoltzmannMethodD2Q9::BoundaryType::CONSTANT, 1),
		m1.getShiftedData(), m1.getShiftedData());
	for (size_t i = 0; i < 2; i++)
	{
		lbm.step(true);
	}
}
#endif  // MAIN_CPP
