#include "LatticeBoltzmannMethod.h"
LatticeBoltzmannMethod::LatticeBoltzmannMethod(quint16 height, quint16 width, QObject *parent): QObject(parent)
{
	Matrix<int> m(10, 10);
	m.print();
	m[{5, 5}] = 42;
	m.print();
	// m.applyFunction(increment);
	// m.print();
	// qDebug() << "-----";
	// m.print();
	// m.shift(Matrix<int>::Direction::E);  // shift to the right
	// m.print();
	// m.shift(Matrix<int>::Direction::S);  // shift down
	// m.print();
	// m.shift(Matrix<int>::Direction::SW); // shift down and to the left
	// m.print();
}
