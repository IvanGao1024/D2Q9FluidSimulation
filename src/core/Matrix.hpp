#ifndef CARTESIAN_MATRIX
#define CARTESIAN_MATRIX

#include <vector>
#include <iostream>
#include <omp.h>

#include "OpenCLMain.hpp"

/**
 * @brief The matrix class with n(Row) x m(Col) dimension.
 *
 * @tparam T
 */
template<typename T>
class Matrix
{
private:
	static const unsigned int MATRIX_DEFAULT_HEIGHT = 1;
	static const unsigned int MATRIX_DEFAULT_WIDTH  = 1;

	unsigned int N;
	unsigned int M;
	unsigned int LENGTH;

	std::vector<T> mData;
	unsigned int   mRowShiftIndex;
	unsigned int   mColShiftIndex;

public:
	/**
	 * @brief Construct a new Cartesian Matrix object with n x m dimensions. Row first then column.
	 *
	 * @param nRow
	 * @param nColumn
	 * @param initialValue
	 */
	Matrix(const unsigned int nRow         = MATRIX_DEFAULT_HEIGHT,
		   const unsigned int nColumn      = MATRIX_DEFAULT_WIDTH,
		   const T            initialValue = T()):
		N(nRow),
		M(nColumn),
		LENGTH(nRow * nColumn)
	{
		mRowShiftIndex = 0;
		mColShiftIndex = 0;
		mData.resize(LENGTH, initialValue);
	}

	/**
	 * @brief Construct a new Cartesian Matrix object with n x m dimensions. Row first then column.
	 *
	 * @param nRow
	 * @param nColumn
	 * @param values
	 * @param magnification
	 */
	Matrix(const unsigned int   nRow,
		   const unsigned int   nColumn,
		   const std::vector<T> values,
		   const unsigned int   magnification = 1):
		N(nRow),
		M(nColumn),
		LENGTH(nRow * nColumn)
	{
		if(values.size() != LENGTH) {
			std::string errMsg = "Inconsistent std::vector length: ";
			errMsg += std::to_string(values.size()) + " vs " + std::to_string(LENGTH);
			throw std::invalid_argument(errMsg);
		}

		mRowShiftIndex = 0;
		mColShiftIndex = 0;
		mData.resize(LENGTH);
#pragma omp parallel for
		for(size_t i = 0; i < values.size(); i++) {
			mData[i] = values.at(i) * magnification;
		}
	}

public:
	Matrix<T>& operator=(const Matrix<T>& rhs)
	{
		if(this != &rhs) {  // Avoid self-assignment
			this->N              = rhs.N;
			this->M              = rhs.M;
			this->LENGTH         = rhs.LENGTH;
			this->mData          = rhs.mData;
			this->mRowShiftIndex = rhs.mRowShiftIndex;
			this->mColShiftIndex = rhs.mColShiftIndex;
		}
		return *this;
	}

	bool operator==(const Matrix<T>& other) const
	{
		if(LENGTH != other.LENGTH || M != other.M || N != other.N) {
			return false;
		}
		for(int i = 0; i < LENGTH; ++i) {
			if(mData[i] != other.mData[i]) {
				return false;
			}
		}
		return true;
	}

	bool operator!=(const Matrix<T>& other) const
	{
		return !(*this == other);
	}

public:  // Helper getter
	unsigned int getWidth() const
	{
		return M;
	}

	unsigned int getHeight() const
	{
		return N;
	}

	unsigned int getLength() const
	{
		return LENGTH;
	}

	unsigned int getRowShiftIndex() const
	{
		return mRowShiftIndex;
	}

	unsigned int getColShiftIndex() const
	{
		return mColShiftIndex;
	}

	std::pair<unsigned int, unsigned int> getShiftIndexPair() const
	{
		return std::pair<unsigned int, unsigned int>(mRowShiftIndex, mColShiftIndex);
	}

	T* getDataData()
	{
		return mData.data();
	}

	// T getData(const unsigned int nRow, const unsigned int nCol)
	// {
	// 	return mData.at(((nRow * N) + (nCol + mColShiftIndex + M) % M - mRowShiftIndex * N + LENGTH) % LENGTH);
	// }

	std::vector<T> getShiftedData(int x = 0, int y = 0) const
	{
		std::vector<T> shiftedData(LENGTH);

		// Calculate the new combined shift indices (x, y) = (nCol, nRow)
		int combinedRowShiftIndex = (mRowShiftIndex + y + N) % N;
		int combinedColShiftIndex = (mColShiftIndex + x + M) % M;

		// #pragma omp parallel for
		for(size_t i = 0; i < LENGTH; ++i) {
			shiftedData[((i - (i % N)) + ((i % N) + combinedColShiftIndex + M) % M - combinedRowShiftIndex * N +
						 LENGTH) %
						LENGTH] = mData[i];
		}
		return shiftedData;
	}

	void shift(int x = 0, int y = 0)
	{
		mRowShiftIndex = (mRowShiftIndex + y + N) % N;
		mColShiftIndex = (mColShiftIndex + x + M) % M;
	}

	void resetData(std::vector<T> data)
	{
		mData          = data;
		mRowShiftIndex = 0;
		mColShiftIndex = 0;
	}

public:
	void fill(const T& value)
	{
#pragma omp parallel for
		for(size_t i = 0; i < mData.size(); ++i) {
			mData[i] = value;
		}
	}

	void indexRevision(const unsigned int nRow, const unsigned int nCol, const T& newValue)
	{
		mData[((nRow * N) + (nCol + mColShiftIndex + M) % M - mRowShiftIndex * N + LENGTH) % LENGTH] = newValue;
	}

	void rowRevision(const unsigned int nRow, const T& newValue)
	{
		validateIndex(0, nRow);

		// Compute the effective shifted row
		unsigned int shiftedRow = (nRow + mRowShiftIndex) % N;

		// The linear index for the first element in the shifted row
		unsigned int shiftedIndex = shiftedRow * M;

#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			// Calculate the index for each element in the shifted row
			unsigned int index = (shiftedIndex + i) % LENGTH;

			// Update the value at the index
			mData[index] = newValue;
		}
	}

	void colRevision(const unsigned int nCol, const T& newValue)
	{
		validateIndex(nCol, 0);

		// Compute the effective shifted column
		unsigned int shiftedCol = (nCol + mColShiftIndex) % M;

#pragma omp parallel for
		for(int row = 0; row < N; ++row) {
			// The linear index for the first element in each row
			unsigned int baseIndex = row * M;

			// Calculate the linear index for each element in the shifted column
			unsigned int index = (baseIndex + shiftedCol) % LENGTH;

			// Update the value at the index
			mData[index] = newValue;
		}
	}

	void print() const
	{
		std::cout << "---------------------- " << M << "x" << N << " ----------------------\n";

		std::vector<T> newVector = getShiftedData(); // Make sure to specify the type of the elements in newVector

		// Initialize variables to keep track of rows and columns
		int row = 0, col = 0;

		// Iterate through the entire shifted data
		for(int i = 0; i < LENGTH; ++i) {
			std::cout << newVector[i];

			// Increment the column counter
			col++;

			// Check if the end of a row has been reached
			if(col == M) {
				std::cout << "\n"; // New line at the end of a row
				col = 0;            // Reset the column counter
			} else {
				std::cout << " | "; // Separator between columns
			}
		}
	}

private:  // Helper
	void validateIndex(int columnX, int rowY) const
	{
		if(columnX < 0 || columnX >= M || rowY < 0 || rowY >= N) {
			std::string message =
				"Index out of bounds: (" + std::to_string(columnX) + ", " + std::to_string(rowY) + ")";
			throw std::out_of_range(message);
		}
	}
};

#endif  // CARTESIAN_MATRIX