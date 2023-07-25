#ifndef MATRIX
#define MATRIX

#include <vector>
#include <thread>
#include <algorithm>
#include <execution>
#include <iostream>

/**
 * @brief The origin is at (0,0), located at the bottom left,
 * with x and y increasing as in a Cartesian coordinate system.
 * 
 * @tparam T 
 */
template<typename T>
class CartesianMatrix
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

	enum Direction { UP, DOWN, LEFT, RIGHT };

public:
	CartesianMatrix(int width, int height, T initialValue = T()): mWidth(width), mHeight(height)
	{
		data.resize(mWidth * mHeight, initialValue);
	}

public:
	bool operator==(const CartesianMatrix<T>& other) const {
		if (mWidth != other.mWidth || mHeight != other.mHeight) {
			return false;
		}
		for (int i = 0; i < mWidth * mHeight; ++i) {
			if (data[i] != other.data[i]) {
				return false;
			}
		}
		return true;
	}

	T& operator[](Index index) {
		validateIndex(index.x, index.y);
		int newX = (base.first + index.x) % mWidth;
		int newY = (base.second + (mHeight - index.y - 1)) % mHeight;
		return data[newY * mWidth + newX];
	}

	T at(Index index) const {
		validateIndex(index.x, index.y);
		int newX = (base.first + index.x + mWidth) % mWidth;
		int newY = (base.second + (mHeight - index.y - 1) + mHeight) % mHeight;
		return data.at(newY * mWidth + newX);
	}

	void batchRevisionX(int x, const T &newValue) {
		validateIndex(x, 0);
		#pragma omp parallel for
		for (int i = 0; i < mHeight; ++i) {
			(*this)[{x, i}] = newValue;
		}
	}

	void batchRevisionY(int y, const T &newValue) {
		validateIndex(0, y);
		#pragma omp parallel for
		for (int i = 0; i < mWidth; ++i) {
			(*this)[{i, y}] = newValue;
		}
	}

    void baseShift(Direction dir) {
        auto [dx, dy] = getShift(dir);
        base.first = (base.first + dx + mWidth) % mWidth;
        base.second = (base.second + dy + mHeight) % mHeight;
        validateIndex(base.first, base.second);
    }

public:  // helper
	void validateIndex(int x, int y) const {
		if(x < 0 || x >= mWidth || y < 0 || y >= mHeight) {
			throw std::out_of_range("Index out of bounds");
		}
	}
	
	std::pair<int, int> getShift(Direction dir) {
		switch(dir) {
		case UP: return {0, 1};
		case DOWN: return {0, -1};
		case LEFT: return {1, 0};
		case RIGHT: return {-1, 0};
		default: return {0, 0};
		}
	}

	void print() {
		std::cout << "---------------------- " << mWidth << "x" << mHeight << " ----------------------\n";
		for(int i = mHeight - 1; i >= 0; --i) {
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