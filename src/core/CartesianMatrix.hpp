#ifndef CARTESIAN_MATRIX
#define CARTESIAN_MATRIX

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
	static const unsigned int MATRIX_DEFAULT_HEIGHT = 1;
	static const unsigned int MATRIX_DEFAULT_WIDTH  = 1;

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
	CartesianMatrix()
	{
		mWidth  = MATRIX_DEFAULT_WIDTH;
		mHeight = MATRIX_DEFAULT_HEIGHT;
		data.resize(mWidth * mHeight, T());
	}

public:
	CartesianMatrix(unsigned int width, unsigned int height, T initialValue = T()): mWidth(width), mHeight(height)
	{
		data.resize(mWidth * mHeight, initialValue);
	}

public:
	CartesianMatrix<T>& operator=(const CartesianMatrix<T>& rhs)
	{
		if(this != &rhs) {  // Avoid self-assignment
			this->base    = rhs.base;
			this->data    = rhs.data;
			this->mWidth  = rhs.mWidth;
			this->mHeight = rhs.mHeight;
		}
		return *this;
	}

	bool operator==(const CartesianMatrix<T>& other) const
	{
		if(mWidth != other.mWidth || mHeight != other.mHeight) {
			return false;
		}
		for(int i = 0; i < mWidth * mHeight; ++i) {
			if(data[i] != other.data[i]) {
				return false;
			}
		}
		return true;
	}

	bool operator!=(const CartesianMatrix<T>& other) const
	{
		return !(*this == other);
	}

	T& operator[](Index index)
	{
		validateIndex(index.x, index.y);
		int newX = (base.first + index.x) % mWidth;
		int newY = (base.second + (mHeight - index.y - 1)) % mHeight;
		return data[newY * mWidth + newX];
	}

	CartesianMatrix<T> operator*(const double& value)
	{
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(int i = 0; i < result.data.size(); ++i)
			result.data[i] *= value;
		return result;
	}

	CartesianMatrix<T> operator+(const CartesianMatrix<T>& rhs)
	{
		if(this->data.size() != rhs.data.size()) {
			throw std::invalid_argument("Dimention mismatch");
		}
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] += rhs.data[i];
		}

		return result;
	}

public:
	T at(Index index) const
	{
		validateIndex(index.x, index.y);
		int newX = (base.first + index.x + mWidth) % mWidth;
		int newY = (base.second + (mHeight - index.y - 1) + mHeight) % mHeight;
		return data.at(newY * mWidth + newX);
	}

	void batchRevisionX(int columnX, const T& newValue)
	{
		validateIndex(columnX, 0);
#pragma omp parallel for
		for(int i = 0; i < mHeight; ++i) {
			(*this)[{columnX, i}] = newValue;
		}
	}

	void batchRevisionY(int rowY, const T& newValue)
	{
		validateIndex(0, rowY);
#pragma omp parallel for
		for(int i = 0; i < mWidth; ++i) {
			(*this)[{i, rowY}] = newValue;
		}
	}

	void baseShift(Direction dir)
	{
		auto [dx, dy] = getShift(dir);
		base.first    = (base.first + dx + mWidth) % mWidth;
		base.second   = (base.second + dy + mHeight) % mHeight;
		validateIndex(base.first, base.second);
	}

	void fill(const T& value)
	{
#pragma omp parallel for
		for(size_t i = 0; i < data.size(); ++i) {
			data[i] = value;
		}
	}

public:  // helper
	unsigned int getWidth() const
	{
		return mWidth;
	}

	unsigned int getHeight() const
	{
		return mHeight;
	}

	void validateIndex(int columnX, int rowY) const
	{
		if(columnX < 0 || columnX >= mWidth || rowY < 0 || rowY >= mHeight) {
			throw std::out_of_range("Index out of bounds");
		}
	}

	std::pair<int, int> getShift(Direction dir)
	{
		switch(dir) {
		case UP: return {0, 1};
		case DOWN: return {0, -1};
		case LEFT: return {1, 0};
		case RIGHT: return {-1, 0};
		default: return {0, 0};
		}
	}

	void print() const
	{
		std::cout << "---------------------- " << mWidth << "x" << mHeight << " ----------------------\n";
		for(int i = mHeight - 1; i >= 0; --i) {
			for(int j = 0; j < mWidth; ++j) {
				std::cout << (*this).at({j, i});
				if(j != mWidth - 1) {
					std::cout << " | ";
				}
			}
			std::cout << '\n';
		}
	}
};

#endif  // CARTESIAN_MATRIX