#ifndef CARTESIAN_MATRIX
#define CARTESIAN_MATRIX

#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include <type_traits>
#include <omp.h>
#include <stdexcept>

#include "OpenCLMain.hpp"

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

public:
	std::pair<int, int> base{0, 0};
	std::vector<T>      data;
	int                 mWidth;
	int                 mHeight;

public:
	struct Index {
		int x, y;
	};

	enum Direction { UP, DOWN, LEFT, RIGHT };


	CartesianMatrix(): mWidth(MATRIX_DEFAULT_WIDTH), mHeight(MATRIX_DEFAULT_HEIGHT)
	{
		data.resize(mWidth * mHeight, T());
	}

	CartesianMatrix(unsigned int nRow, unsigned int nColumn, T initialValue = T()): mWidth(nColumn), mHeight(nRow)
	{
		data.resize(mWidth * mHeight, initialValue);
	}

	CartesianMatrix(unsigned int nRow, unsigned int nColumn, const std::vector<std::vector<T>>& values):
		mWidth(nColumn),
		mHeight(nRow)
	{
		if(values.empty()) {
			data.resize(mWidth * mHeight, T());
		} else {
			if(values.size() != nRow) {
				throw std::out_of_range("Row count mismatch");
			}

			// Use a shared exception pointer to capture the first exception thrown.
			std::exception_ptr excPtr = nullptr;

#pragma omp parallel for
			for(size_t i = 0; i < values.size(); i++) {
				try {
					if(values[i].size() != nColumn) {
						throw std::out_of_range("Column count mismatch for a specific row");
					}
				} catch(...) {
// Store exception and break from the loop
#pragma omp critical
					{
						if(!excPtr) {
							excPtr = std::current_exception();
						}
					}
				}
			}

			// Rethrow the exception outside the parallel region
			if(excPtr) {
				std::rethrow_exception(excPtr);
			}

			data.resize(mWidth * mHeight);

#pragma omp parallel for collapse(2)
			for(size_t i = 0; i < mHeight; ++i) {
				for(size_t j = 0; j < mWidth; ++j) {
					data[i * mWidth + j] = values[i][j];
				}
			}
		}
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

	// Scalar multiplication function
	template<typename Scalar, typename = typename std::enable_if<std::is_arithmetic<Scalar>::value, Scalar>::type>
	CartesianMatrix<T> operator+(const Scalar& value) const
	{
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] += value;
		}
		return result;
	}

	CartesianMatrix<T> operator+(const CartesianMatrix<T>& rhs)
	{
		if(this->data.size() != rhs.data.size()) {
			throw std::out_of_range("Dimention mismatch");
		}
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] += rhs.data[i];
		}

		return result;
	}

	// Scalar multiplication function
	template<typename Scalar, typename = typename std::enable_if<std::is_arithmetic<Scalar>::value, Scalar>::type>
	CartesianMatrix<T> operator-(const Scalar& value) const
	{
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] -= value;
		}
		return result;
	}

	CartesianMatrix<T> operator-(const CartesianMatrix<T>& rhs)
	{
		if(this->data.size() != rhs.data.size()) {
			throw std::out_of_range("Dimention mismatch");
		}
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] -= rhs.data[i];
		}

		return result;
	}

	// Scalar multiplication function
	template<typename Scalar, typename = typename std::enable_if<std::is_arithmetic<Scalar>::value, Scalar>::type>
	CartesianMatrix<T> operator*(const Scalar& value) const
	{
		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] *= value;
		}
		return result;
	}

	// Matrix multiplication function
	CartesianMatrix<T> operator*(const CartesianMatrix<T>& other)
	{
		if(mWidth != other.mHeight) {
			throw std::out_of_range(
				"Dimension mismatch: The number of rows in the first matrix must equal the number of columns in the second.");
		}

		CartesianMatrix<T> result(mHeight,
								  other.mWidth);  // Adjusting the size based on the usual matrix multiplication rules

#pragma omp parallel for collapse(2)
		for(int y = 0; y < mHeight; ++y) {
			for(int x = 0; x < other.mWidth; ++x) {
				T sum = 0;
				for(int k = 0; k < mWidth; ++k) {
					sum += (*this).at({k, mHeight - 1 - y}) * other.at({x, other.mHeight - 1 - k});
				}
				result[{x, mHeight - 1 - y}] = sum;  // Using Cartesian coordinates
			}
		}

		return result;
	}

	// Matrix divided by a scalar
	template<typename Scalar, typename = typename std::enable_if<std::is_arithmetic<Scalar>::value, Scalar>::type>
	CartesianMatrix<T> operator/(const Scalar& value) const
	{
		if(value == 0) {
			throw std::runtime_error("Division by zero is undefined.");
		}

		CartesianMatrix<T> result(*this);
#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			result.data[i] /= value;
		}
		return result;
	}

	// Scalar divided by a matrix
	template<typename Scalar, typename = typename std::enable_if<std::is_arithmetic<Scalar>::value, Scalar>::type>
	friend CartesianMatrix<T> operator/(const Scalar& scalarValue, const CartesianMatrix<T>& matrix)
	{
		CartesianMatrix<T> result(matrix);

#pragma omp parallel for
		for(size_t i = 0; i < result.data.size(); ++i) {
			if(result.data[i] == 0) {
				throw std::runtime_error("Matrix contains zero elements, making division by matrix undefined.");
			}
			result.data[i] = scalarValue / result.data[i];
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

	void fillRandom()
	{
		static_assert(std::is_arithmetic<T>::value, "Template type T must be numeric");

		std::random_device rd;
		std::mt19937       gen(rd());

		if constexpr(std::is_integral<T>::value) {
			std::uniform_int_distribution<T> dist(0, 2147483647);

#pragma omp parallel for
			for(size_t i = 0; i < data.size(); ++i) {
				data[i] = dist(gen);
			}
		} else if constexpr(std::is_floating_point<T>::value) {
			std::uniform_real_distribution<T> dist(0, 1);

#pragma omp parallel for
			for(size_t i = 0; i < data.size(); ++i) {
				data[i] = dist(gen);
			}
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
			std::string message =
				"Index out of bounds: (" + std::to_string(columnX) + ", " + std::to_string(rowY) + ")";
			throw std::out_of_range(message);
		}
	}

	std::pair<int, int> getShift(Direction dir) const
	{
		switch(dir) {
		case UP: return {0, 1};
		case DOWN: return {0, -1};
		case LEFT: return {1, 0};
		case RIGHT: return {-1, 0};
		default: throw std::invalid_argument("Unknown direction passed to getShift.");
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