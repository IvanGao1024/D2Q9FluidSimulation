#ifndef MATRIX
#define MATRIX

#include <vector>
#include <thread>
#include <algorithm>
#include <execution>
#include <iostream>
using namespace std;

template<typename T>
class Matrix
{
private:
	std::pair<int, int> base{0, 0};
	std::vector<T>      data;
	int                 mWidth;
	int                 mHeight;

public:
	struct Index {
		int x, y;
	};

	enum Direction { N, NE, E, SE, S, SW, W, NW };

public:
	Matrix(int width, int height, T initialValue = T()): mWidth(width), mHeight(height)
	{
		data.resize(mWidth * mHeight, initialValue);
	}

	T& operator[](Index index)
	{
		int newX = (base.first + index.x + mWidth) % mWidth;
		int newY = (base.second + index.y + mHeight) % mHeight;
		return data[newY * mWidth + newX];
	}

	T at(Index index) const
	{
		int newX = (base.first + index.x + mWidth) % mWidth;
		int newY = (base.second + index.y + mHeight) % mHeight;
		return data.at(newY * mWidth + newX);
	}

	// void applyFunction(void(*func)(T&)) {
	//     std::for_each(std::execution::par, data.begin(), data.end(), func);
	// }

	void baseShift(Direction dir)
	{
		auto [dx, dy] = getShift(dir);
		base.first    = (base.first + dx + mWidth) % mWidth;
		base.second   = (base.second + dy + mHeight) % mHeight;
	}

    void rowRevision(int row, const T & newValue) {
        
    }

    void columnRevision(int column, const T & newValue) {
        
    }

public:  // helper
	std::pair<int, int> getShift(Direction dir)
	{
		switch(dir) {
		case N: return {0, -1};
		case NE: return {1, -1};
		case E: return {1, 0};
		case SE: return {1, 1};
		case S: return {0, 1};
		case SW: return {-1, 1};
		case W: return {-1, 0};
		case NW: return {-1, -1};
		default: return {0, 0};
		}
	}

	void print()
	{
		std::cout << "---------------------- " << mWidth << "x" << mHeight << " ----------------------\n";
		for(int i = 0; i < mHeight; ++i) {
			for(int j = 0; j < mWidth; ++j) {
				std::cout << (*this)[{j, i}];
				if(j != mWidth - 1) {
					std::cout << " | ";
				}
			}
			std::cout << '\n';
		}
	}
};

#endif  // Matrix