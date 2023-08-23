#ifndef CARTESIAN_MATRIX
#define CARTESIAN_MATRIX

#include <vector>
#include <iostream>
#include <omp.h>

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

	std::vector<T> mData;
	unsigned int   mShiftIndex;
	unsigned int   mWidth;
	unsigned int   mHeight;
	unsigned int   mLength;

public:
	struct Index {
		int x, y;
	};

	/**
	 * @brief Construct a new Cartesian Matrix object with n x m dimensions. Row first then column.
	 *
	 * @param nRow
	 * @param nColumn
	 * @param initialValue
	 */
	CartesianMatrix(const unsigned int nRow         = MATRIX_DEFAULT_HEIGHT,
					const unsigned int nColumn      = MATRIX_DEFAULT_WIDTH,
					const T            initialValue = T()):
		mHeight(nRow),
		mWidth(nColumn),
		mLength(nRow * nColumn)
	{
		mShiftIndex = 0;
		mData.resize(mLength, initialValue);
	}

	/**
	 * @brief Construct a new Cartesian Matrix object with n x m dimensions. Row first then column.
	 *
	 * @param nRow
	 * @param nColumn
	 * @param values
	 * @param magnification
	 */
	CartesianMatrix(const unsigned int   nRow,
					const unsigned int   nColumn,
					const std::vector<T> values,
					const unsigned int   magnification = 1):
		mHeight(nRow),
		mWidth(nColumn),
		mLength(nRow * nColumn)
	{
		if(values.size() != mLength) {
			std::string errMsg = "Inconsistent std::vector length: ";
			errMsg += std::to_string(values.size()) + " vs " + std::to_string(mLength);
			throw std::invalid_argument(errMsg);
		}

		mShiftIndex = 0;
		mData.resize(mLength);
#pragma omp parallel for
		for(size_t i = 0; i < values.size(); i++) {
			mData[i] = values.at(i) * magnification;
		}
	}

public:
	CartesianMatrix<T>& operator=(const CartesianMatrix<T>& rhs)
	{
		if(this != &rhs) {  // Avoid self-assignment
			this->mData       = rhs.mData;
			this->mShiftIndex = rhs.mShiftIndex;
			this->mWidth      = rhs.mWidth;
			this->mHeight     = rhs.mHeight;
			this->mLength     = rhs.mLength;
		}
		return *this;
	}

	bool operator==(const CartesianMatrix<T>& other) const
	{
		if(mLength != other.mLength || mWidth != other.mWidth || mHeight != other.mHeight) {
			return false;
		}
		for(int i = 0; i < mLength; ++i) {
			if(mData[i] != other.mData[i]) {
				return false;
			}
		}
		return true;
	}

	bool operator!=(const CartesianMatrix<T>& other) const
	{
		return !(*this == other);
	}

public:  // Helper getter
	unsigned int getWidth() const
	{
		return mWidth;
	}

	unsigned int getHeight() const
	{
		return mHeight;
	}

	unsigned int getLength() const
	{
		return mLength;
	}

	unsigned int getShiftIndex() const
	{
		return mShiftIndex;
	}

	std::vector<T> getShiftedData()
	{
		if(mShiftIndex == 0) {
			return mData;
		} else {
			std::vector<T> shiftedData(mLength);

#pragma omp parallel for
			for(size_t i = 0; i < mLength; ++i) {
				shiftedData[i] = mData[(i + mShiftIndex) % mLength];
			}
			return shiftedData;
		}
	}

	T* getDataData()
	{
		return mData.data();
	}

public:
	void shiftUp(unsigned int nRow = 1)
	{
		if(nRow < 0 || nRow >= mHeight)  // >= mHeight, as shifting by mHeight would be invalid.
		{
			throw std::invalid_argument("Invalid number of row shift up.");
		} else {
			mShiftIndex = (mShiftIndex + mWidth * nRow) % mLength;  // Multiplied by nRow to actually shift by nRow.
		}
	}

	void resetData(std::vector<T> data)
	{
		mShiftIndex = 0;
		mData       = data;
	}

public:
	void fill(const T& value)
	{
#pragma omp parallel for
		for(size_t i = 0; i < mData.size(); ++i) {
			mData[i] = value;
		}
	}

	void rowRevision(unsigned int nRow, const T& newValue)
	{
		validateIndex(0, nRow);

		// Calculate the linear index for the shifted row
		int shiftedIndex = ((nRow * mWidth) + mShiftIndex) % mLength;

#pragma omp parallel for
		for(int i = 0; i < mWidth; ++i) {
			// Update the value at the shifted row index
			mData[(shiftedIndex + i) % mLength] = newValue;
		}
	}

	void colRevision(unsigned int nCol, const T& newValue)
	{
		validateIndex(nCol, 0);  // Note: We're validating a column, not a row, so nCol should come first.

#pragma omp parallel for
		for(int row = 0; row < mHeight; ++row) {
			// Calculate the original linear index based on row and column
			int linearIndex = row * mWidth + nCol;

			// Compute the shifted linear index
			int shiftedIndex = (linearIndex + mShiftIndex) % mLength;

			// Update the value at the shifted index
			mData[shiftedIndex] = newValue;
		}
	}

	void print() const
	{
		std::cout << "---------------------- " << mWidth << "x" << mHeight << " ----------------------\n";

		// Iterate through each row, starting from the bottom
		for(int row = 0; row < mHeight; ++row) {
			// Iterate through each column
			for(int col = 0; col < mWidth; ++col) {
				// Calculate the original linear index based on row and column
				int linearIndex = row * mWidth + col;

				// Adjust for the row shift
				int shiftedRow = (row + mShiftIndex / mWidth) % mHeight;

				// Compute the shifted linear index
				int shiftedIndex = shiftedRow * mWidth + col;

				std::cout << mData.at(shiftedIndex);

				// Separator
				if(col != mWidth - 1) {
					std::cout << " | ";
				}
			}
			std::cout << '\n';  // End of row
		}
	}

private:  // Helper
	void validateIndex(int columnX, int rowY) const
	{
		if(columnX < 0 || columnX >= mWidth || rowY < 0 || rowY >= mHeight) {
			std::string message =
				"Index out of bounds: (" + std::to_string(columnX) + ", " + std::to_string(rowY) + ")";
			throw std::out_of_range(message);
		}
	}
};

#endif  // CARTESIAN_MATRIX