#include "D2Q9.h"
D2Q9::D2Q9(quint16 height, quint16 width, QObject *parent): QObject(parent)
{
    CartesianMatrix<float> m[9];
    for (int i = 0; i < 9; ++i) { 
        mData[i] = CartesianMatrix<float>(width + 1, height + 1, 0.0F);
    }
}