#ifndef OPENCL_MAIN
#define OPENCL_MAIN

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <variant>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>
#include <map>

#include <cassert>
#include <memory>
#include <iostream>
#include <cstdarg>  // For va_start, va_end
#include <set>
#include <stdexcept>  // For std::invalid_argument
#include <regex>

#include "Matrix.hpp"
class OpenCLMain
{
private:
	struct MachineProfile {
		std::string mPlatformName;
		std::string mDeviceName;
		size_t      mOptimalWorkGroupSize;
	};

private:
	static inline MachineProfile UserMachineProfile;

	static inline cl::Platform         mPlatform;
	static inline cl::Device           mDevice;
	static inline cl::NDRange          mLocal;
	static inline cl::Context          mContext;
	static inline cl::Program::Sources mArithmeticSources;
	static inline cl::Program          mArithmeticProgram;

	// parameter
	static inline cl::CommandQueue        mQueue;
	static inline cl::NDRange             mGlobal;
	static inline std::set<char>          mAvailableCacheIndex;
	static inline char                    mNewCacheIndex;
	static inline unsigned int            mArrayLength;
	static inline std::vector<cl::Buffer> mBuffers;

private:
	OpenCLMain()
	{
		// Handel Platforms
		std::vector<cl::Platform> all_platforms;
		try {
			// Retrieve platforms
			cl::Platform::get(&all_platforms);
		} catch(const std::exception& ex) {
			std::cerr << "[OpenCL] Error retrieving platforms: " << ex.what() << std::endl;
			throw;
		}

		// If no platform found
		if(all_platforms.empty()) {
			std::cerr << "[OpenCL] No platforms found!" << std::endl;
			throw std::runtime_error("[OpenCL] Platform not found.");
		}

		// List found platforms
		// for(const cl::Platform& platform : all_platforms) {
		// 	std::cout << "[OpenCL] Platform found: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
		// }

		// Select the default platform
		mPlatform                        = all_platforms[0];
		UserMachineProfile.mPlatformName = mPlatform.getInfo<CL_PLATFORM_NAME>();
		// std::cout << "[OpenCL] Platform selected: " << UserMachineProfile.mPlatformName << "\n";

		// Handel Devices
		std::vector<cl::Device> all_devices;
		mPlatform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

		// If no device found
		if(all_devices.empty()) {
			std::cerr << "[OpenCL] No device found under" << UserMachineProfile.mPlatformName << "\n";
			throw std::runtime_error("[OpenCL] Device not found.");
		}

		// List found devices
		// for(const cl::Device& device : all_devices) {
		// 	std::cout << "[OpenCL] Device found: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		// }

		// Select the default device
		mDevice                        = all_devices[0];
		UserMachineProfile.mDeviceName = mDevice.getInfo<CL_DEVICE_NAME>();
		// std::cout << "[OpenCL] Device selected:" << UserMachineProfile.mDeviceName << "\n";

		// Set local work group size
		mDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &UserMachineProfile.mOptimalWorkGroupSize);
		cl::NDRange mLocalWorkGroupSize(UserMachineProfile.mOptimalWorkGroupSize);
		// std::cout << "[OpenCL] Local work-group size set to:" << UserMachineProfile.mOptimalWorkGroupSize << "\n";

		// Handel context
		mContext = cl::Context({mDevice});

		// Initiate Arithmetic Kernel
		std::string arithmeticKernelCode = R"(
			void kernel kernelAddingArray(global double* C, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, global const double* B, const unsigned int shiftBRow, const unsigned int shiftBCol, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				unsigned int originalIndexB = (((i - i % M) / M + shiftBRow) %  N) * M + (i - shiftBCol + M) % M;
				C[i] = A[originalIndexA] + B[originalIndexB];
			}
			void kernel kernelAddingConstant(global double* B, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, const double C, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				B[i] = A[originalIndexA] + C;
			}

			void kernel kernelSubtractingArray(global double* C, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, global const double* B, const unsigned int shiftBRow, const unsigned int shiftBCol, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				unsigned int originalIndexB = (((i - i % M) / M + shiftBRow) %  N) * M + (i - shiftBCol + M) % M;
				C[i] = A[originalIndexA] - B[originalIndexB];
			}
			void kernel kernelSubtractingConstant(global double* B, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, const double C, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				B[i] = A[originalIndexA] - C;
			}
			void kernel kernelConstantSubtracting(global double* B, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, const double C, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				B[i] = C - A[originalIndexA];
			}

			void kernel kernelMultiplicatingArray(global double* C, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, global const double* B, const unsigned int shiftBRow, const unsigned int shiftBCol, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				unsigned int originalIndexB = (((i - i % M) / M + shiftBRow) %  N) * M + (i - shiftBCol + M) % M;
				C[i] = A[originalIndexA] * B[originalIndexB];
			}
			void kernel kernelMultiplicatingConstant(global double* B, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, const double C, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				B[i] = A[originalIndexA] * C;
			}

			void kernel kernelDividingByArray(global double* C, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, global const double* B, const unsigned int shiftBRow, const unsigned int shiftBCol, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				unsigned int originalIndexB = (((i - i % M) / M + shiftBRow) %  N) * M + (i - shiftBCol + M) % M;
				double bValue = B[originalIndexB];
				if (bValue != 0) {  // Ensure don't divide by zero
					C[i] = A[originalIndexA] / bValue;
				} else {
					C[i] = 0;
				}
			}
			void kernel kernelDividingByConstant(global double* B, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, const double C, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				if (C != 0) {  // Ensure don't divide by zero
					unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
					B[i] = A[originalIndexA] / C;
				} else {
					B[i] = 0;
				}
			}
			void kernel kernelConstantDividingBy(global double* B, global const double* A, const unsigned int shiftARow, const unsigned int shiftACol, const double C, const unsigned int N, const unsigned int M) {
				unsigned int i = get_global_id(0);
				unsigned int originalIndexA = (((i - i % M) / M + shiftARow) %  N) * M + (i - shiftACol + M) % M;
				double aValue = A[originalIndexA];
				if (aValue != 0) {  // Ensure don't divide by zero
					B[i] = C / aValue;
				} else {
					B[i] = 0;
				}
			}
		)";
		mArithmeticSources.push_back({arithmeticKernelCode.c_str(), arithmeticKernelCode.length()});
		mArithmeticProgram = cl::Program(mContext, mArithmeticSources);
		if(mArithmeticProgram.build({mDevice}) != CL_SUCCESS) {
			std::cout << " Error building: " << mArithmeticProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevice) << "\n";
			throw std::runtime_error("Error building arithmetic source code");
		}
	}

	~OpenCLMain()
	{
		// Cleanup OpenCL resources
		mArithmeticProgram = nullptr;
	}

	// Delete copy/move constructors and assignment operators
	OpenCLMain(OpenCLMain const&)     = delete;
	void operator=(OpenCLMain const&) = delete;
	OpenCLMain(OpenCLMain&&)          = delete;
	void operator=(OpenCLMain&&)      = delete;

public:
	static OpenCLMain& instance()
	{
		static OpenCLMain instance;  // Guaranteed to be destroyed and instantiated on first use
		return instance;
	}

	// Convert an infix expression to postfix using the shunting yard algorithm
	static std::queue<std::string> enqueueArithmeticFormula(const std::string& expression)
	{
		std::queue<std::string> output;
		std::stack<char>        operators;
		std::string             token;

		auto isOperator = [](char c) -> bool {
			return c == '+' || c == '-' || c == '*' || c == '/';
		};

		std::unordered_map<char, int> precedence = {{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}};

		for(char ch : expression) {
			if(isspace(ch)) {
				if(!token.empty()) {
					output.push(token);
					token.clear();
				}
			} else if(isOperator(ch)) {
				if(!token.empty()) {
					output.push(token);
					token.clear();
				}

				while(!operators.empty() && isOperator(operators.top()) &&
					  precedence[ch] <= precedence[operators.top()]) {
					output.push(std::string(1, operators.top()));
					operators.pop();
				}
				operators.push(ch);
			} else if(ch == '(') {
				operators.push(ch);
			} else if(ch == ')') {
				if(!token.empty()) {
					output.push(token);
					token.clear();
				}

				while(!operators.empty() && operators.top() != '(') {
					output.push(std::string(1, operators.top()));
					operators.pop();
				}

				if(!operators.empty()) {  // Pop the '(' from the stack
					operators.pop();
				}
			} else {  // Assume it's part of an operand (could be multi-digit numbers or multi-letter variable names)
				token += ch;
			}
		}

		// If any remaining token is left, push it to the output
		if(!token.empty()) {
			output.push(token);
		}

		// Push any remaining operators to the output
		while(!operators.empty()) {
			output.push(std::string(1, operators.top()));
			operators.pop();
		}

		return output;
	}

	/**
	 * @brief
	 * @attention a character after last character used will be used for result, so max 25 variable can be used.
	 * @attention The use of 'A'-'Y' as variable name must be used in sequencial order.
	 * @tparam T
	 * @param expression
	 * @param arrayLength
	 * @param values
	 */
	static Matrix<double> evaluateArithmeticFormula(
		const std::string&                  expression,
		const std::vector<Matrix<double>*>& array = std::vector<Matrix<double>*>())
	{
		unsigned int mArrayN = 0;
		unsigned int mArrayM = 0;

		if(array.size() != 0) {
			mArrayN = array.at(0)->getN();
			mArrayM = array.at(0)->getM();

			for(size_t i = 0; i < array.size(); i++) {
				if(array.at(i)->getN() != mArrayN || array.at(i)->getM() != mArrayM) {
					throw std::invalid_argument("Given matrix's dimension mismatch.");
				}
			}
		}
		mArrayLength = mArrayN * mArrayM;

		char refIndex = 'A';
		// Check for number of variable mismatch by counting unique uppercase characters
		std::set<char> uniqueUppercaseChars;
		for(char ch : expression) {
			if(isupper(ch)) {
				if(ch == refIndex) {
					uniqueUppercaseChars.insert(ch);
					refIndex = refIndex + 1;
				} else if(ch == (refIndex - 1)) {
					continue;
				} else {
					throw std::invalid_argument("Variable Reference are not in alphabatic order.");
				}
			}
		}
		if(uniqueUppercaseChars.size() != array.size()) {
			throw std::invalid_argument(
				"Error: Mismatch between the number of variable used in expression and the number of variable given.");
		}

		// Initialize kernels
		auto kernelAddingArray =
			cl::compatibility::make_kernel<cl::Buffer,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   unsigned int,
										   unsigned int>(cl::Kernel(mArithmeticProgram, "kernelAddingArray"));
		auto kernelAddingConstant = cl::compatibility::
			make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int, double, unsigned int, unsigned int>(
				cl::Kernel(mArithmeticProgram, "kernelAddingConstant"));
		auto kernelSubtractingArray =
			cl::compatibility::make_kernel<cl::Buffer,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   unsigned int,
										   unsigned int>(cl::Kernel(mArithmeticProgram, "kernelSubtractingArray"));
		auto kernelSubtractingConstant = cl::compatibility::
			make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int, double, unsigned int, unsigned int>(
				cl::Kernel(mArithmeticProgram, "kernelSubtractingConstant"));
		auto kernelConstantSubtracting = cl::compatibility::
			make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int, double, unsigned int, unsigned int>(
				cl::Kernel(mArithmeticProgram, "kernelConstantSubtracting"));
		auto kernelMultiplicatingArray =
			cl::compatibility::make_kernel<cl::Buffer,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   unsigned int,
										   unsigned int>(cl::Kernel(mArithmeticProgram, "kernelMultiplicatingArray"));
		auto kernelMultiplicatingConstant = cl::compatibility::
			make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int, double, unsigned int, unsigned int>(
				cl::Kernel(mArithmeticProgram, "kernelMultiplicatingConstant"));
		auto kernelDividingByArray =
			cl::compatibility::make_kernel<cl::Buffer,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   cl::Buffer,
										   unsigned int,
										   unsigned int,
										   unsigned int,
										   unsigned int>(cl::Kernel(mArithmeticProgram, "kernelDividingByArray"));
		auto kernelDividingByConstant = cl::compatibility::
			make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int, double, unsigned int, unsigned int>(
				cl::Kernel(mArithmeticProgram, "kernelDividingByConstant"));
		auto kernelConstantDividingBy = cl::compatibility::
			make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int, double, unsigned int, unsigned int>(
				cl::Kernel(mArithmeticProgram, "kernelConstantDividingBy"));

		// Initialize parameter
		if(array.size() != 0) {
			mQueue   = cl::CommandQueue(mContext, mDevice);
			mGlobal  = cl::NDRange(mArrayLength);
			mBuffers = std::vector<cl::Buffer>(array.size() + 1);
			// Allocate buffer memory
#pragma omp parallel for
			for(size_t i = 0; i < array.size(); i++) {
				mBuffers[i] = cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(double) * mArrayLength);
			}
			mBuffers[array.size()] = cl::Buffer(mContext, CL_MEM_WRITE_ONLY, sizeof(double) * mArrayLength);
			// std::cout << "Total of " << array.size() + 1 << " buffer created.\n";

			// Initialize buffer
#pragma omp parallel for
			for(size_t i = 0; i < array.size(); i++) {
				mQueue.enqueueWriteBuffer(mBuffers[i],
										  CL_TRUE,
										  0,
										  sizeof(double) * mArrayLength,
										  array[i]->getDataData());
			}
		}

		// set for tracking cache index
		mAvailableCacheIndex.clear();
		mAvailableCacheIndex.insert('A' + array.size());
		mNewCacheIndex = 'A' + array.size() + 1;

		// Forming the post fix queue
		std::queue<std::string> mPostfixNotationQueue = enqueueArithmeticFormula(expression);

		// Stack for evaluation
		std::stack<std::variant<double, char>> evalStack;

		// Loop over stack and evaluate
		while(!mPostfixNotationQueue.empty()) {
			std::string token = mPostfixNotationQueue.front();
			mPostfixNotationQueue.pop();
			if(isDouble(token)) {
				// std::cout << "Push " << token << "\n";
				evalStack.push(std::stod(token));
			} else if(token.size() == 1 && isupper(token[0])) {
				// std::cout << "Push " << token << "\n";
				evalStack.push(token[0]);
			} else if(token == "+") {
				std::variant<double, char> second = evalStack.top();
				evalStack.pop();
				std::variant<double, char> first = evalStack.top();
				evalStack.pop();
				// std::visit([&first, &second](auto firstValue) {
				// 	std::visit([&firstValue, &second](auto secondValue) {
				// 		std::cout << "Caculate " << firstValue << " + " << secondValue << "\n";
				// 	}, second);
				// }, first);

				if(std::holds_alternative<double>(first) && std::holds_alternative<double>(second)) {
					evalStack.push(std::get<double>(first) + std::get<double>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<double>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(first) - 'A';
					kernelAddingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										 mBuffers[cacheChar - 'A'],
										 mBuffers[charIndex],
										 charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
										 charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
										 std::get<double>(second),
										 mArrayN,
										 mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<double>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(second) - 'A';
					kernelAddingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										 mBuffers[cacheChar - 'A'],
										 mBuffers[charIndex],
										 charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
										 charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
										 std::get<double>(first),
										 mArrayN,
										 mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar  = getCacheIndex();
					unsigned int charIndex1 = std::get<char>(first) - 'A';
					unsigned int charIndex2 = std::get<char>(second) - 'A';
					kernelAddingArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
									  mBuffers[cacheChar - 'A'],
									  mBuffers[charIndex1],
									  charIndex1 < array.size() ? array.at(charIndex1)->getRowShiftIndex() : 0,
									  charIndex1 < array.size() ? array.at(charIndex1)->getColShiftIndex() : 0,
									  mBuffers[charIndex2],
									  charIndex2 < array.size() ? array.at(charIndex2)->getRowShiftIndex() : 0,
									  charIndex2 < array.size() ? array.at(charIndex2)->getColShiftIndex() : 0,
									  mArrayN,
									  mArrayM)
						.wait();
					if(charIndex1 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(charIndex2 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				}
			} else if(token == "-") {
				std::variant<double, char> second = evalStack.top();
				evalStack.pop();
				std::variant<double, char> first = evalStack.top();
				evalStack.pop();
				// std::visit([&first, &second](auto firstValue) {
				// 	std::visit([&firstValue, &second](auto secondValue) {
				// 		std::cout << "Caculate " << firstValue << " - " << secondValue << "\n";
				// 	}, second);
				// }, first);

				if(std::holds_alternative<double>(first) && std::holds_alternative<double>(second)) {
					evalStack.push(std::get<double>(first) - std::get<double>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<double>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(first) - 'A';
					kernelSubtractingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											  mBuffers[cacheChar - 'A'],
											  mBuffers[charIndex],
											  charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
											  charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
											  std::get<double>(second),
											  mArrayN,
											  mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<double>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(second) - 'A';
					kernelConstantSubtracting(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											  mBuffers[cacheChar - 'A'],
											  mBuffers[charIndex],
											  charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
											  charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
											  std::get<double>(first),
											  mArrayN,
											  mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar  = getCacheIndex();
					unsigned int charIndex1 = std::get<char>(first) - 'A';
					unsigned int charIndex2 = std::get<char>(second) - 'A';
					kernelSubtractingArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										   mBuffers[cacheChar - 'A'],
										   mBuffers[charIndex1],
										   charIndex1 < array.size() ? array.at(charIndex1)->getRowShiftIndex() : 0,
										   charIndex1 < array.size() ? array.at(charIndex1)->getColShiftIndex() : 0,
										   mBuffers[charIndex2],
										   charIndex2 < array.size() ? array.at(charIndex2)->getRowShiftIndex() : 0,
										   charIndex2 < array.size() ? array.at(charIndex2)->getColShiftIndex() : 0,
										   mArrayN,
										   mArrayM)
						.wait();
					if(charIndex1 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(charIndex2 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				}
			} else if(token == "*") {
				std::variant<double, char> second = evalStack.top();
				evalStack.pop();
				std::variant<double, char> first = evalStack.top();
				evalStack.pop();
				// std::visit([&first, &second](auto firstValue) {
				// 	std::visit([&firstValue, &second](auto secondValue) {
				// 		std::cout << "Caculate " << firstValue << " * " << secondValue << "\n";
				// 	}, second);
				// }, first);

				if(std::holds_alternative<double>(first) && std::holds_alternative<double>(second)) {
					evalStack.push(std::get<double>(first) * std::get<double>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<double>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(first) - 'A';
					kernelMultiplicatingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
												 mBuffers[cacheChar - 'A'],
												 mBuffers[charIndex],
												 charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
												 charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
												 std::get<double>(second),
												 mArrayN,
												 mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<double>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(second) - 'A';
					kernelMultiplicatingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
												 mBuffers[cacheChar - 'A'],
												 mBuffers[charIndex],
												 charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
												 charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
												 std::get<double>(first),
												 mArrayN,
												 mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar  = getCacheIndex();
					unsigned int charIndex1 = std::get<char>(first) - 'A';
					unsigned int charIndex2 = std::get<char>(second) - 'A';
					kernelMultiplicatingArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											  mBuffers[cacheChar - 'A'],
											  mBuffers[charIndex1],
											  charIndex1 < array.size() ? array.at(charIndex1)->getRowShiftIndex() : 0,
											  charIndex1 < array.size() ? array.at(charIndex1)->getColShiftIndex() : 0,
											  mBuffers[charIndex2],
											  charIndex2 < array.size() ? array.at(charIndex2)->getRowShiftIndex() : 0,
											  charIndex2 < array.size() ? array.at(charIndex2)->getColShiftIndex() : 0,
											  mArrayN,
											  mArrayM)
						.wait();
					if(charIndex1 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(charIndex2 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				}
			} else if(token == "/") {
				std::variant<double, char> second = evalStack.top();
				evalStack.pop();
				std::variant<double, char> first = evalStack.top();
				evalStack.pop();
				// std::visit([&first, &second](auto firstValue) {
				// 	std::visit([&firstValue, &second](auto secondValue) {
				// 		std::cout << "Caculate " << firstValue << " / " << secondValue << "\n";
				// 	}, second);
				// }, first);

				if(std::holds_alternative<double>(first) && std::holds_alternative<double>(second)) {
					evalStack.push(std::get<double>(first) / std::get<double>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<double>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(first) - 'A';
					kernelDividingByConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											 mBuffers[cacheChar - 'A'],
											 mBuffers[charIndex],
											 charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
											 charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
											 std::get<double>(second),
											 mArrayN,
											 mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<double>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar = getCacheIndex();
					unsigned int charIndex = std::get<char>(second) - 'A';
					kernelConstantDividingBy(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											 mBuffers[cacheChar - 'A'],
											 mBuffers[charIndex],
											 charIndex < array.size() ? array.at(charIndex)->getRowShiftIndex() : 0,
											 charIndex < array.size() ? array.at(charIndex)->getColShiftIndex() : 0,
											 std::get<double>(first),
											 mArrayN,
											 mArrayM)
						.wait();
					if(charIndex >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char         cacheChar  = getCacheIndex();
					unsigned int charIndex1 = std::get<char>(first) - 'A';
					unsigned int charIndex2 = std::get<char>(second) - 'A';
					kernelDividingByArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										  mBuffers[cacheChar - 'A'],
										  mBuffers[charIndex1],
										  charIndex1 < array.size() ? array.at(charIndex1)->getRowShiftIndex() : 0,
										  charIndex1 < array.size() ? array.at(charIndex1)->getColShiftIndex() : 0,
										  mBuffers[charIndex2],
										  charIndex2 < array.size() ? array.at(charIndex2)->getRowShiftIndex() : 0,
										  charIndex2 < array.size() ? array.at(charIndex2)->getColShiftIndex() : 0,
										  mArrayN,
										  mArrayM)
						.wait();
					if(charIndex1 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(charIndex2 >= array.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(cacheChar);
				}
			} else {
				std::cout << "unknown token :" << token << std::endl;
			}

			// std::vector<T>          resultVector(1);
			// std::variant<int, char> result = evalStack.top();
			// if(std::holds_alternative<char>(result)) {
			// 	std::cout << "ref " << std::get<char>(result);
			// 	resultVector.resize(mArrayLength);
			// 	mQueue.enqueueReadBuffer(mBuffers[std::get<char>(result) - 'A'],
			// 							 CL_TRUE,
			// 							 0,
			// 							 sizeof(T) * mArrayLength,
			// 							 resultVector.data());
			// } else {
			// 	resultVector[0] = std::get<int>(result);
			// }
			// std::cout << " Stack top " << resultVector[0] << "\n";
		}

		// Final result
		if(!evalStack.empty()) {
			Matrix<double>             result;
			std::variant<double, char> resultVal = evalStack.top();
			if(std::holds_alternative<char>(resultVal)) {
				result = Matrix<double>(mArrayN, mArrayM);
				mQueue.enqueueReadBuffer(mBuffers[std::get<char>(resultVal) - 'A'],
										 CL_TRUE,
										 0,
										 sizeof(double) * mArrayLength,
										 result.getDataData());
			} else {
				result = Matrix<double>(1, 1, std::vector<double>{std::get<double>(resultVal)});
			}
			return result;
		} else {
			throw std::runtime_error("Unknown error, result stack empty.");
		}
	}

private:
	static bool isDouble(const std::string& token)
	{
		try {
			std::stod(token);
			return true;
		} catch(const std::invalid_argument&) {
			return false;
		} catch(const std::out_of_range&) {
			return false;
		}
	}

	static char getCacheIndex()
	{
		// std::cout << "Pool has ";
		// for(auto ch: mAvailableCacheIndex) {
		// 	std::cout << ch;
		// }
		// std::cout << std::endl;

		char index;
		if(!mAvailableCacheIndex.empty()) {
			index = *mAvailableCacheIndex.begin();
			mAvailableCacheIndex.erase(mAvailableCacheIndex.begin());
		} else {
			index          = mNewCacheIndex;
			mNewCacheIndex = mNewCacheIndex + 1;
			mBuffers.push_back(cl::Buffer(mContext, CL_MEM_WRITE_ONLY, sizeof(double) * mArrayLength));
			// std::cout << static_cast<char>(mNewCacheIndex - 1) << " allocated. new buffer allocated.\n";
		}
		// std::cout << index << " allocated.\n";
		return index;
	}
};

#endif  // OPENCL_MAIN