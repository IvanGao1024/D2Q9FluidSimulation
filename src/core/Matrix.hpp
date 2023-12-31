#ifndef MATRIX
#define MATRIX

#include <vector>
#include <iostream>
#include <omp.h>

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
	Matrix(const unsigned int nRow          = MATRIX_DEFAULT_HEIGHT,
		   const unsigned int nColumn       = MATRIX_DEFAULT_WIDTH,
		   const T            initialValue  = T(),
		   const unsigned int rowShiftIndex = 0,
		   const unsigned int colShiftIndex = 0):
		N(nRow),
		M(nColumn),
		LENGTH(nRow * nColumn)
	{
		mRowShiftIndex = rowShiftIndex;
		mColShiftIndex = colShiftIndex;
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
		   const double         magnification = 1,
		   const unsigned int   rowShiftIndex = 0,
		   const unsigned int   colShiftIndex = 0):
		N(nRow),
		M(nColumn),
		LENGTH(nRow * nColumn)
	{
		if(values.size() != LENGTH) {
			std::string errMsg = "Inconsistent std::vector length: ";
			errMsg += std::to_string(values.size()) + " vs " + std::to_string(LENGTH);
			throw std::invalid_argument(errMsg);
		}

		mRowShiftIndex = rowShiftIndex;
		mColShiftIndex = colShiftIndex;
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
	unsigned int getM() const
	{
		return M;
	}

	unsigned int getN() const
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

	T getValue(const unsigned int index) const
	{
		return mData.at((((index - index % M) / M + mRowShiftIndex) % N) * M + (index - mColShiftIndex + M) % M);
	}

	std::vector<T> getData() const
	{
		return mData;
	}

	T* getDataData()
	{
		return mData.data();
	}

	void resetData(std::vector<T> data)
	{
		mData          = data;
		mRowShiftIndex = 0;
		mColShiftIndex = 0;
	}

	std::vector<T> getShiftedData(int x = 0, int y = 0) const
	{
		std::vector<T> shiftedData(LENGTH);

		// Calculate the new combined shift indices (x, y) = (nCol, nRow)
		int combinedRowShiftIndex = (mRowShiftIndex + y + N) % N;
		int combinedColShiftIndex = (mColShiftIndex + x + M) % M;

#pragma omp parallel for
		for(size_t i = 0; i < LENGTH; ++i) {
			shiftedData[i] =
				mData[(((i - i % M) / M + combinedRowShiftIndex) % N) * M + (i - combinedColShiftIndex + M) % M];
		}
		return shiftedData;
	}

	void shift(int x = 0, int y = 0)
	{
		mRowShiftIndex = (mRowShiftIndex + y + N) % N;
		mColShiftIndex = (mColShiftIndex + x + M) % M;
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
		validateIndex(nCol, nRow);
		unsigned int i                                                                     = nCol + nRow * M;
		mData[(((i - i % M) / M + mRowShiftIndex) % N) * M + (i - mColShiftIndex + M) % M] = newValue;
	}

	void rowRevision(const unsigned int nRow, const T& newValue)
	{
		validateIndex(0, nRow);
#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			unsigned int index = nRow * M + i;
			mData[(((index - index % M) / M + mRowShiftIndex) % N) * M + (index - mColShiftIndex + M) % M] = newValue;
		}
	}

	void colRevision(const unsigned int nCol, const T& newValue)
	{
		validateIndex(nCol, 0);
#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			unsigned int index = i * M + nCol;
			mData[(((index - index % M) / M + mRowShiftIndex) % N) * M + (index - mColShiftIndex + M) % M] = newValue;
		}
	}

	void print() const
	{
		std::cout << "---------------------- " << M << "x" << N << " ----------------------\n";
		std::vector<T> newVector = getShiftedData();
		int row = 0, col = 0;
		for(int i = 0; i < LENGTH; ++i) {
			std::cout << newVector[i];

			col++;

			if(col == M) {
				std::cout << "\n";
				col = 0;
			} else {
				std::cout << " | ";
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

public:
	void topAdiabatic()
	{
#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			unsigned int topIndex      = i;
			unsigned int topBelowIndex = i + M;
			unsigned int originalIndexTop =
				(((topIndex - topIndex % M) / M + mRowShiftIndex) % N) * M + (topIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexTopBelow = (((topBelowIndex - topBelowIndex % M) / M + mRowShiftIndex) % N) * M +
												 (topBelowIndex - mColShiftIndex + M) % M;
			mData[originalIndexTop] = mData[originalIndexTopBelow];
		}
	}

	void bottomAdiabatic()
	{
#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			unsigned int bottomIndex         = i + M * (N - 1);
			unsigned int bottomAboveIndex    = i + M * (N - 2);
			unsigned int originalIndexBottom = (((bottomIndex - bottomIndex % M) / M + mRowShiftIndex) % N) * M +
											   (bottomIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexbottomAbove =
				(((bottomAboveIndex - bottomAboveIndex % M) / M + mRowShiftIndex) % N) * M +
				(bottomAboveIndex - mColShiftIndex + M) % M;
			mData[originalIndexBottom] = mData[originalIndexbottomAbove];
		}
	}

	void leftAdiabatic()
	{
#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
			unsigned int leftIndex      = i * M;
			unsigned int leftRightIndex = i * M + 1;
			unsigned int originalIndexLeft =
				(((leftIndex - leftIndex % M) / M + mRowShiftIndex) % N) * M + (leftIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexLeftRight =
				(((leftRightIndex - leftRightIndex % M) / M + mRowShiftIndex) % N) * M +
				(leftRightIndex - mColShiftIndex + M) % M;
			mData[originalIndexLeft] = mData[originalIndexLeftRight];
		}
	}

	void rightAdiabatic()
	{
#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
			unsigned int rightIndex     = (i + 1) * M - 1;
			unsigned int rightLeftIndex = (i + 1) * M - 2;
			unsigned int originalIndexRight =
				(((rightIndex - rightIndex % M) / M + mRowShiftIndex) % N) * M + (rightIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexRightLeft =
				(((rightLeftIndex - rightLeftIndex % M) / M + mRowShiftIndex) % N) * M +
				(rightLeftIndex - mColShiftIndex + M) % M;
			mData[originalIndexRight] = mData[originalIndexRightLeft];
		}
	}

	void topDirichlet(const double C, const Matrix<T>& matrix)
	{
		std::vector<T> other              = matrix.getData();
		unsigned int   otherRowShiftIndex = matrix.getRowShiftIndex();
		unsigned int   otherColShiftIndex = matrix.getColShiftIndex();
#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			unsigned int topIndex = i;
			unsigned int originalIndexTop =
				(((topIndex - topIndex % M) / M + mRowShiftIndex) % N) * M + (topIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexOtherTop = (((topIndex - topIndex % M) / M + otherRowShiftIndex) % N) * M +
												 (topIndex - otherColShiftIndex + M) % M;
			mData[originalIndexTop] = C - other[originalIndexOtherTop];
		}
	}

	void bottomDirichlet(const double C, const Matrix<T>& matrix)
	{
		std::vector<T> other              = matrix.getData();
		unsigned int   otherRowShiftIndex = matrix.getRowShiftIndex();
		unsigned int   otherColShiftIndex = matrix.getColShiftIndex();
#pragma omp parallel for
		for(int i = 0; i < M; ++i) {
			unsigned int bottomIndex         = i + M * (N - 1);
			unsigned int originalIndexBottom = (((bottomIndex - bottomIndex % M) / M + mRowShiftIndex) % N) * M +
											   (bottomIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexOtherBottom =
				(((bottomIndex - bottomIndex % M) / M + otherRowShiftIndex) % N) * M +
				(bottomIndex - otherColShiftIndex + M) % M;
			mData[originalIndexBottom] = C - other[originalIndexOtherBottom];
		}
	}

	void leftDirichlet(const double C, const Matrix<T>& matrix)
	{
		std::vector<T> other              = matrix.getData();
		unsigned int   otherRowShiftIndex = matrix.getRowShiftIndex();
		unsigned int   otherColShiftIndex = matrix.getColShiftIndex();
#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
			unsigned int leftIndex = i * M;
			unsigned int originalIndexLeft =
				(((leftIndex - leftIndex % M) / M + mRowShiftIndex) % N) * M + (leftIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexOtherLeft = (((leftIndex - leftIndex % M) / M + otherRowShiftIndex) % N) * M +
												  (leftIndex - otherColShiftIndex + M) % M;
			mData[originalIndexLeft] = C - other[originalIndexOtherLeft];
		}
	}

	void rightDirichlet(const double C, const Matrix<T>& matrix)
	{
		std::vector<T> other              = matrix.getData();
		unsigned int   otherRowShiftIndex = matrix.getRowShiftIndex();
		unsigned int   otherColShiftIndex = matrix.getColShiftIndex();
#pragma omp parallel for
		for(int i = 0; i < N; ++i) {
			unsigned int rightIndex = i * M + N - 1;
			unsigned int originalIndexRight =
				(((rightIndex - rightIndex % M) / M + mRowShiftIndex) % N) * M + (rightIndex - mColShiftIndex + M) % M;
			unsigned int originalIndexOtherRight = (((rightIndex - rightIndex % M) / M + otherRowShiftIndex) % N) * M +
												   (rightIndex - otherColShiftIndex + M) % M;
			mData[originalIndexRight] = C - other[originalIndexOtherRight];
		}
	}
};

#endif  // MATRIX