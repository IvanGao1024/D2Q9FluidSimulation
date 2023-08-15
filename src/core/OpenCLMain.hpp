#ifndef OPENCL_MAIN
#define OPENCL_MAIN

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

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
		size_t mOptimalWorkGroupSize;
	};

private:
	static inline MachineProfile UserMachineProfile;

	static inline cl::Platform         mPlatform;
	static inline cl::Device           mDevice;
	static inline cl::NDRange 		   mLocal;
	static inline cl::Context          mContext;
	static inline cl::Program::Sources mSources;
	static inline cl::Program          mArithmeticProgram;

private:
	static inline const std::unordered_map<char, int> precedence = {
		{'+', 1},
		{'-', 1},
		{'*', 2},
		{'/', 2},
		{'^', 3}
	};

    static bool isOperator(char c) {
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
		for(const cl::Platform& platform : all_platforms) {
			std::cout << "[OpenCL] Platform found: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
		}

		// Select the default platform
		mPlatform                        = all_platforms[0];
		UserMachineProfile.mPlatformName = mPlatform.getInfo<CL_PLATFORM_NAME>();
		std::cout << "[OpenCL] Platform selected: " << UserMachineProfile.mPlatformName << "\n";

		// Handel Devices
		std::vector<cl::Device> all_devices;
		mPlatform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

		// If no device found
		if(all_devices.empty()) {
			std::cerr << "[OpenCL] No device found under" << UserMachineProfile.mPlatformName << "\n";
			throw std::runtime_error("[OpenCL] Device not found.");
		}

		// List found devices
		for(const cl::Device& device : all_devices) {
			std::cout << "[OpenCL] Device found: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		}

		// Select the default device
		mDevice                        = all_devices[0];
		UserMachineProfile.mDeviceName = mDevice.getInfo<CL_DEVICE_NAME>();
		std::cout << "[OpenCL] Device selected:" << UserMachineProfile.mDeviceName << "\n";

		// Set local work group size
		mDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &UserMachineProfile.mOptimalWorkGroupSize);
		cl::NDRange mLocalWorkGroupSize(UserMachineProfile.mOptimalWorkGroupSize);
		std::cout << "[OpenCL] Local work-group size set to:" << UserMachineProfile.mOptimalWorkGroupSize << "\n";

		// Handel context
		mContext = cl::Context({mDevice});

		// Initiate Arithmetic Kernel
		std::string kernelCode = R"(
			void kernel kernelAddingArray(global const unsigned int* A, global const unsigned int* B, global unsigned int* C) {
				C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
			}
			void kernel kernelSubtractingArray(global const unsigned int* A, global const unsigned int* B, global unsigned int* C) {
				C[get_global_id(0)] = A[get_global_id(0)] - B[get_global_id(0)];
			}
			void kernel kernelMultiplicatingArray(global const unsigned int* A, global const unsigned int* B, global unsigned int* C) {
				C[get_global_id(0)] = A[get_global_id(0)] * B[get_global_id(0)];
			}
			void kernel kernelDividingByArray(global const unsigned int* A, global const unsigned int* B, global unsigned int* C) {
				unsigned int bValue = B[get_global_id(0)];
				if (bValue != 0) {  // Ensure don't divide by zero
					C[get_global_id(0)] = A[get_global_id(0)] / bValue;
				} else {
					C[get_global_id(0)] = 0;
				}
			}
			void kernel kernelAddingConstant(global const unsigned int* A, global unsigned int* B, const unsigned int C) {
				B[get_global_id(0)] = A[get_global_id(0)] + C;
			}
			void kernel kernelSubtractingConstant(global const unsigned int* A, global unsigned int* B, const unsigned int C) {
				B[get_global_id(0)] = A[get_global_id(0)] - C;
			}
			void kernel kernelMultiplicatingConstant(global const unsigned int* A, global unsigned int* B, const unsigned int C) {
				B[get_global_id(0)] = A[get_global_id(0)] * C;
			}
			void kernel kernelDividedByConstant(global const unsigned int* A, global unsigned int* B, const unsigned int C) {
				if (C != 0) {  // Ensure don't divide by zero
					B[get_global_id(0)] = A[get_global_id(0)] / C;
				} else {
					B[get_global_id(0)] = 0;
				}
			}
			void kernel kernelOneOverArray(global const unsigned int* A, global unsigned int* B) {
				unsigned int value = A[get_global_id(0)];
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
	static OpenCLMain& instance() {
        static OpenCLMain instance; // Guaranteed to be destroyed and instantiated on first use
        return instance;
	}

	// Convert an infix expression to postfix using the shunting yard algorithm
	static std::queue<std::string> enqueueArithmeticFormula(const std::string &expression) {
		std::queue<std::string> output;
		std::stack<char> operators;
		std::string token;

		for (char ch : expression) {
			if (isspace(ch)) {
				if (!token.empty()) {
					output.push(token);
					token.clear();
				}
			} else if (isOperator(ch)) {
				if (!token.empty()) {
					output.push(token);
					token.clear();
				}

				while (!operators.empty() && isOperator(operators.top()) && precedence.at(ch) <= precedence.at(operators.top())) {
					output.push(std::string(1, operators.top()));
					operators.pop();
				}
				operators.push(ch);
			} else if (ch == '(') {
				operators.push(ch);
			} else if (ch == ')') {
				if (!token.empty()) {
					output.push(token);
					token.clear();
				}

				while (!operators.empty() && operators.top() != '(') {
					output.push(std::string(1, operators.top()));
					operators.pop();
				}

				if (!operators.empty()) {  // Pop the '(' from the stack
					operators.pop();
				}
			} else {  // Assume it's part of an operand (could be multi-digit numbers or multi-letter variable names)
				token += ch;
			}
		}

		// If any remaining token is left, push it to the output
		if (!token.empty()) {
			output.push(token);
		}

		// Push any remaining operators to the output
		while (!operators.empty()) {
			output.push(std::string(1, operators.top()));
			operators.pop();
		}

		return output;
	}

    template<typename... Args>
    static void evaluateArithmeticFormula(const std::string &expression, Args... args) {
		if (sizeof...(args) > 26) {
			throw std::invalid_argument("Error: Too many arguments passed!");
		}

        // Count unique uppercase characters
        std::set<char> uniqueUppercaseChars;
        for (char ch : expression) {
            if (isupper(ch)) {
                uniqueUppercaseChars.insert(ch);
            }
        }

		if (uniqueUppercaseChars.size() != sizeof...(args)) {
			throw std::invalid_argument("Mismatch between number of variables in expression and provided arguments.");
		}
		
		// Start evaluating the queue
        std::queue<std::string> mPostfixNotationQueue = enqueueArithmeticFormula(expression);
        
        // Map to store variable values like A=1, B=2, etc.
        std::map<char, double> values;
        
        char var = 'A';
        auto assign = [&](double value) {
            values[var++] = value;
        };

        // Assign passed values to variables
        (assign(args), ...);

        // Stack for evaluation
        std::stack<double> evalStack;

        while(!mPostfixNotationQueue.empty()) {
			std::string token = mPostfixNotationQueue.front();
			mPostfixNotationQueue.pop();

			if(token == "+") {
				double second = evalStack.top(); evalStack.pop();
				double first = evalStack.top(); evalStack.pop();
				std::cout << first << " + " << second << std::endl;
				evalStack.push(0); // stub result
			}
			else if(token == "-") {
				double second = evalStack.top(); evalStack.pop();
				double first = evalStack.top(); evalStack.pop();
				std::cout << first << " - " << second << std::endl;
				evalStack.push(0); // stub result
			}
			else if(token == "*") {
				double second = evalStack.top(); evalStack.pop();
				double first = evalStack.top(); evalStack.pop();
				std::cout << first << " * " << second << std::endl;
				evalStack.push(0); // stub result
			}
			else if(token == "/") {
				double second = evalStack.top(); evalStack.pop();
				double first = evalStack.top(); evalStack.pop();
				std::cout << first << " / " << second << std::endl;
				evalStack.push(0); // stub result
			}
			else if(token.size() == 1 && isupper(token[0])) {
				std::cout << "Variable " << token << " with value " << values[token[0]] << std::endl;
				evalStack.push(values[token[0]]);
			}
			else if(std::regex_match(token, std::regex("[0-9]+"))) {  // Check if the token matches the pattern for positive integers
				int value = std::stoi(token);  // Convert the token to an integer
				std::cout << "Integer: " << value << std::endl;
				evalStack.push(value);
			}
			else {
				std::cout << "unknown token :" << token << std::endl;
			}
		}


        // Final result (for now just a stub)
        if (!evalStack.empty()) {
            std::cout << "Final result: " << evalStack.top() << std::endl;
        }
    }

	// static inline auto kernelAddingArray;

	// static inline cl::CommandQueue     mQueue;
	// static inline cl::NDRange global;

	// static inline cl::Buffer A_d;
	// static inline cl::Buffer B_d;
	// static inline cl::Buffer C_d;

		// A_d = cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 10000*10000);
		// B_d = cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 10000*10000);
		// C_d = cl::Buffer(mContext, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * 10000*10000);
		// mK1 = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(
        // 	cl::Kernel(mArithmeticProgram, "kernelAddingArray")
    	// );

		// mK2 = cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, unsigned int>(
        // 	cl::Kernel(mArithmeticProgram, "kernelAddingConstant")
    	// );

		// Handel queue
		// mQueue = cl::CommandQueue(mContext, mDevice);
		// // assert(A_h.size() == B_h.size());
		// // assert(B_h.size() == C_h.size());
		// // assert(A_h.size() == C_h.size());

		// const unsigned int SIZE = size;
		// global = cl::NDRange(SIZE);

		// // Example cont'd
		// mQueue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(unsigned int) * SIZE, A_h);
		// mQueue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(unsigned int) * SIZE, B_h);
		
		// (*pArrayAdditionKernel)(cl::EnqueueArgs(mQueue, global, local), A_d, B_d, C_d).wait();
        
		// arrayAdditionKernel(cl::EnqueueArgs(mQueue, global, local), A_d, B_d, C_d).wait();
        
		// Retrieve data
		// mQueue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

		// for (size_t i = 0; i < 20; i++)
		// {
		// 	std::cout << C_h[i];
		// }
		// std::cout << "\n";
	

	~OpenCLMain()
	{	

		// // Cleanup OpenCL resources
		// clReleaseMemObject(buffer);
		// clReleaseKernel(kernel);
		// clReleaseProgram(program);
		// clReleaseCommandQueue(queue);
		// clReleaseContext(context);
	}
	
	// Delete copy/move constructors and assignment operators
	OpenCLMain(OpenCLMain const&) = delete;
	void operator=(OpenCLMain const&) = delete;
	OpenCLMain(OpenCLMain&&) = delete;
	void operator=(OpenCLMain&&) = delete;
};

#endif  // OPENCL_MAIN