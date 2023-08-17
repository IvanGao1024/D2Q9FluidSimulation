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
	static inline cl::Program::Sources mSources;
	static inline cl::Program          mArithmeticProgram;

	// parameter
	static inline cl::CommandQueue        mQueue;
	static inline cl::NDRange             mGlobal;
	static inline unsigned int            mArrayLength;
	static inline std::vector<cl::Buffer> mBuffers;
	static inline std::set<char>          mAvailableCacheIndex;
	static inline char                    mNewCacheIndex;

private:
	static inline const std::unordered_map<char, int> precedence = {{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'^', 3}};

	static bool isOperator(char c)
	{
		return precedence.find(c) != precedence.end();
	}

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
		std::string kernelCode = R"(
			void kernel kernelAddingArray(global const int* A, global const int* B, global int* C) {
				C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
			}
			void kernel kernelAddingConstant(global const int* A, global int* B, const int C) {
				B[get_global_id(0)] = A[get_global_id(0)] + C;
			}

			void kernel kernelSubtractingArray(global const unsigned int* A, global const unsigned int* B, global unsigned int* C) {
				C[get_global_id(0)] = A[get_global_id(0)] - B[get_global_id(0)];
			}
			void kernel kernelSubtractingConstant(global const unsigned int* A, global unsigned int* B, const unsigned int C) {
				B[get_global_id(0)] = A[get_global_id(0)] - C;
			}
			void kernel kernelConstantSubtracting(global const unsigned int* A, global unsigned int* B, const unsigned int C) {
				B[get_global_id(0)] = C - A[get_global_id(0)];
			}

			void kernel kernelMultiplicatingArray(global const int* A, global const int* B, global int* C) {
				C[get_global_id(0)] = A[get_global_id(0)] * B[get_global_id(0)];
			}
			void kernel kernelMultiplicatingConstant(global const int* A, global int* B, const int C) {
				B[get_global_id(0)] = A[get_global_id(0)] * C;
			}

			void kernel kernelDividingByArray(global const int* A, global const int* B, global int* C) {
				int bValue = B[get_global_id(0)];
				if (bValue != 0) {  // Ensure don't divide by zero
					C[get_global_id(0)] = A[get_global_id(0)] / bValue;
				} else {
					C[get_global_id(0)] = 0;
				}
			}
			void kernel kernelDividedByConstant(global const int* A, global int* B, const int C) {
				if (C != 0) {  // Ensure don't divide by zero
					B[get_global_id(0)] = A[get_global_id(0)] / C;
				} else {
					B[get_global_id(0)] = 0;
				}
			}
			void kernel kernelOneOverArray(global const int* A, global int* B) {
				int value = A[get_global_id(0)];
				if (value != 0) {  // Ensure don't divide by zero
					B[get_global_id(0)] = 1 / value;
				} else {
					B[get_global_id(0)] = 0;
				}
			}
		)";
		mSources.push_back({kernelCode.c_str(), kernelCode.length()});
		mArithmeticProgram = cl::Program(mContext, mSources);
		if(mArithmeticProgram.build({mDevice}) != CL_SUCCESS) {
			std::cout << " Error building: " << mArithmeticProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevice) << "\n";
			throw std::runtime_error("Error building source code");
		}
	}

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
					  precedence.at(ch) <= precedence.at(operators.top())) {
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

		// std::queue<std::string> copy = output;
		// std::string             result;
		// while(!copy.empty()) {
		// 	result += copy.front() + " ";
		// 	copy.pop();
		// }
		// std::cout << "Postfix queue " << result << "\n";
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
	template<typename T>
	static std::vector<T> evaluateArithmeticFormula(const std::string& expression,
													unsigned int       arrayLength = 0,
													std::vector<T*>    arrayValues = std::vector<T*>())
	{
		// std::cout << "Evaluate Arithmetic Formula: " << expression << "\n";
		mArrayLength = arrayLength;

		if(!std::is_arithmetic<T>::value) {
			throw std::invalid_argument("Error: vector type is not numeric.");
		}

		// Check for number of variable mismatch by counting unique uppercase characters
		std::set<char> uniqueUppercaseChars;
		for(char ch : expression) {
			if(isupper(ch)) {
				uniqueUppercaseChars.insert(ch);
			}
		}
		// std::cout << "Total of " << uniqueUppercaseChars.size() << " variables detected.\n";
		// std::cout << "Total of " << arrayValues.size() << " arrayValues given.\n";
		if(uniqueUppercaseChars.size() != arrayValues.size()) {
			throw std::invalid_argument(
				"Error: Mismatch between the number of variable used in expression and the number of variable given.");
		}

		// Initialize kernels
		auto kernelAddingArray = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(
			cl::Kernel(mArithmeticProgram, "kernelAddingArray"));
		auto kernelAddingConstant = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
			cl::Kernel(mArithmeticProgram, "kernelAddingConstant"));
		auto kernelSubtractingArray = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(
			cl::Kernel(mArithmeticProgram, "kernelSubtractingArray"));
		auto kernelSubtractingConstant = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
			cl::Kernel(mArithmeticProgram, "kernelSubtractingConstant"));
		auto kernelConstantSubtracting = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
			cl::Kernel(mArithmeticProgram, "kernelConstantSubtracting"));
		auto kernelMultiplicatingArray = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(
			cl::Kernel(mArithmeticProgram, "kernelMultiplicatingArray"));
		auto kernelMultiplicatingConstant = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
			cl::Kernel(mArithmeticProgram, "kernelMultiplicatingConstant"));

		// Initialize parameter
		mQueue  = cl::CommandQueue(mContext, mDevice);
		mGlobal = cl::NDRange(mArrayLength);
		if(arrayValues.size() != 0) {
			mBuffers = std::vector<cl::Buffer>(arrayValues.size() + 1);
			// Allocate buffer memory
			for(size_t i = 0; i < arrayValues.size(); i++) {
				mBuffers[i] = cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(T) * mArrayLength);
			}
			mBuffers[arrayValues.size()] = cl::Buffer(mContext, CL_MEM_WRITE_ONLY, sizeof(T) * mArrayLength);
			// std::cout << "Total of " << arrayValues.size() + 1 << " buffer created.\n";

			// Initialize buffer
			for(size_t i = 0; i < arrayValues.size(); i++) {
				mQueue.enqueueWriteBuffer(mBuffers[i], CL_TRUE, 0, sizeof(T) * mArrayLength, arrayValues[i]);
			}
		} else {
			// std::cout << "Trivial Arithmetic Formula. No buffer or kernel initialized.\n";
		}

		// set for tracking cache index
		mAvailableCacheIndex.clear();
		mAvailableCacheIndex.insert('A' + arrayValues.size());
		mNewCacheIndex = 'A' + arrayValues.size() + 1;

		// Forming the post fix queue
		std::queue<std::string> mPostfixNotationQueue = enqueueArithmeticFormula(expression);

		// Stack for evaluation
		std::stack<std::variant<int, char>> evalStack;

		// Loop over stack and evaluate
		while(!mPostfixNotationQueue.empty()) {
			std::string token = mPostfixNotationQueue.front();
			mPostfixNotationQueue.pop();
			// std::cout << "token:" << token << "\n";
			if(std::regex_match(token, std::regex("-?[0-9]+"))) {
				evalStack.push(std::stoi(token));
			} else if(token.size() == 1 && isupper(token[0])) {
				evalStack.push(token[0]);
			} else if(token == "+") {
				std::variant<int, char> second = evalStack.top();
				evalStack.pop();
				std::variant<int, char> first = evalStack.top();
				evalStack.pop();

				if(std::holds_alternative<int>(first) && std::holds_alternative<int>(second)) {
					evalStack.push(std::get<int>(first) + std::get<int>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<int>(second)) {
					char index = getCacheIndex<T>();
					kernelAddingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										 mBuffers[std::get<char>(first) - 'A'],
										 mBuffers[index - 'A'],
										 std::get<int>(second));
					if(std::get<char>(first) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(index);
				} else if(std::holds_alternative<int>(first) && std::holds_alternative<char>(second)) {
					char index = getCacheIndex<T>();
					kernelAddingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										 mBuffers[std::get<char>(second) - 'A'],
										 mBuffers[index - 'A'],
										 std::get<int>(first))
						.wait();
					if(std::get<char>(second) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(index);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char index = getCacheIndex<T>();
					kernelAddingArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
									  mBuffers[std::get<char>(first) - 'A'],
									  mBuffers[std::get<char>(second) - 'A'],
									  mBuffers[index - 'A'])
						.wait();
					if(std::get<char>(first) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(std::get<char>(second) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(index);
				}
			} else if(token == "-") {
				std::variant<int, char> second = evalStack.top();
				evalStack.pop();
				std::variant<int, char> first = evalStack.top();
				evalStack.pop();

				if(std::holds_alternative<int>(first) && std::holds_alternative<int>(second)) {
					evalStack.push(std::get<int>(first) - std::get<int>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<int>(second)) {
					char index = getCacheIndex<T>();
					kernelSubtractingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											  mBuffers[std::get<char>(first) - 'A'],
											  mBuffers[index - 'A'],
											  std::get<int>(second))
						.wait();
					if(std::get<char>(first) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(index);
				} else if(std::holds_alternative<int>(first) && std::holds_alternative<char>(second)) {
					char index = getCacheIndex<T>();
					kernelConstantSubtracting(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											  mBuffers[std::get<char>(second) - 'A'],
											  mBuffers[index - 'A'],
											  std::get<int>(first))
						.wait();
					if(std::get<char>(second) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(index);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char index = getCacheIndex<T>();
					kernelSubtractingArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
										   mBuffers[std::get<char>(first) - 'A'],
										   mBuffers[std::get<char>(second) - 'A'],
										   mBuffers[index - 'A'])
						.wait();
					if(std::get<char>(first) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(std::get<char>(second) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(index);
				}
			} else if(token == "*") {
				std::variant<int, char> second = evalStack.top();
				evalStack.pop();
				std::variant<int, char> first = evalStack.top();
				evalStack.pop();

				if(std::holds_alternative<int>(first) && std::holds_alternative<int>(second)) {
					evalStack.push(std::get<int>(first) * std::get<int>(second));
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<int>(second)) {
					char index = getCacheIndex<T>();
					kernelMultiplicatingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
												 mBuffers[std::get<char>(first) - 'A'],
												 mBuffers[index - 'A'],
												 std::get<int>(second))
						.wait();
					if(std::get<char>(first) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					evalStack.push(index);
				} else if(std::holds_alternative<int>(first) && std::holds_alternative<char>(second)) {
					char index = getCacheIndex<T>();
					kernelMultiplicatingConstant(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
												 mBuffers[std::get<char>(second) - 'A'],
												 mBuffers[index - 'A'],
												 std::get<int>(first))
						.wait();
					if(std::get<char>(second) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(index);
				} else if(std::holds_alternative<char>(first) && std::holds_alternative<char>(second)) {
					char index = getCacheIndex<T>();
					kernelMultiplicatingArray(cl::EnqueueArgs(mQueue, mGlobal, mLocal),
											  mBuffers[std::get<char>(first) - 'A'],
											  mBuffers[std::get<char>(second) - 'A'],
											  mBuffers[index - 'A'])
						.wait();
					if(std::get<char>(first) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(first));
					}
					if(std::get<char>(second) >= arrayValues.size())  // meaning is a cache index
					{
						mAvailableCacheIndex.insert(std::get<char>(second));
					}
					evalStack.push(index);
				}
			}
			// else if(token == "/") {
			// 	double second = evalStack.top(); evalStack.pop();
			// 	double first = evalStack.top(); evalStack.pop();
			// 	std::cout << first << " / " << second << std::endl;
			// 	evalStack.push(0); // stub result
			// }
			// else if(token.size() == 1 && isupper(token[0])) {
			// 	std::cout << "Variable " << token << " with value " << values[token[0]] << std::endl;
			// 	evalStack.push(values[token[0]]);
			// }
			// else if() {  // Check if the token matches the pattern for positive integers
			// 	int value = std::stoi(token);  // Convert the token to an integer
			// 	std::cout << "Integer: " << value << std::endl;
			// 	evalStack.push(value);
			// }
			else {
				std::cout << "unknown token :" << token << std::endl;
			}
		}

		// Final result
		std::vector<T> resultVector(1);
		if(!evalStack.empty()) {
			std::variant<int, char> result = evalStack.top();
			if(std::holds_alternative<char>(result)) {
				resultVector.resize(mArrayLength);
				mQueue.enqueueReadBuffer(mBuffers[std::get<char>(result) - 'A'],
										 CL_TRUE,
										 0,
										 sizeof(T) * mArrayLength,
										 resultVector.data());
			} else {
				resultVector[0] = std::get<int>(result);
			}
			// std::cout << "Evaluate Arithmetic Formula: " << expression << " Finished with result " << resultVector[0]
			// 		  << "\n";
			return resultVector;
		} else {
			throw std::runtime_error("Unknown error, result stack empty or unknown reference to variable.");
		}
	}

	template<typename T>
	static char getCacheIndex()
	{
		char index;
		if(!mAvailableCacheIndex.empty()) {
			index = *mAvailableCacheIndex.begin();
			mAvailableCacheIndex.erase(mAvailableCacheIndex.begin());
		} else {
			index = mNewCacheIndex;
			mAvailableCacheIndex.insert(mNewCacheIndex);
			mNewCacheIndex = mNewCacheIndex + 1;
			mBuffers.push_back(cl::Buffer(mContext, CL_MEM_WRITE_ONLY, sizeof(T) * mArrayLength));
		}
		return index;
	}
	// Retrieve data
	// mQueue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);
	// static inline cl::Buffer A_d;
	// static inline cl::Buffer B_d;
	// static inline cl::Buffer C_d;

	// // Example cont'd
	// for (size_t i = 0; i < 20; i++)
	// {
	// 	std::cout << C_h[i];
	// }
	// std::cout << "\n";
	// clReleaseMemObject(buffer);
	// clReleaseKernel(kernel);
	// clReleaseCommandQueue(queue);

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
};

#endif  // OPENCL_MAIN