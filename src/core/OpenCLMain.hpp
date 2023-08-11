#ifndef OPENCL_MAIN
#define OPENCL_MAIN

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <vector>
#include <iostream>

// inline const char* KERNEL_SRC = R"CLC(
// __kernel void addOne(__global float* matrix, const unsigned int count) {
//     int idx = get_global_id(0);
//     if (idx < count) {
//         matrix[idx] += 1.0f;
//     }
// }
// )CLC";

class OpenCLMain
{
	struct MachineProfile {
		std::string mPlatformName;
		std::string mDeviceName;
	};

private:
	MachineProfile UserMachineProfile;

	cl::Platform         mPlatform;
	cl::Device           mDevice;
	cl::Context          mContext;
	cl::CommandQueue     mQueue;
	cl::Program::Sources mSources;
	cl::Program          mProgram;

public:
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

		// Handel context
		mContext = cl::Context({mDevice});

		// Example
		const int SIZE = 10;
		// h for host, d for device
		int A_h[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		int B_h[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
		int C_h[SIZE];

		cl::Buffer A_d(mContext, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
		cl::Buffer B_d(mContext, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
		cl::Buffer C_d(mContext, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE);

		// Handel queue
		mQueue = cl::CommandQueue(mContext, mDevice);

		// Example cont'd
		mQueue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
		mQueue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int) * SIZE, B_h);
		std::string kernelCode = R"(
            void kernel simple_add(global const int* A, global const int* B, global int* C){
                C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];
            }
        )";

		mSources.push_back({kernelCode.c_str(), kernelCode.length()});
		mProgram = cl::Program(mContext, mSources);

		if(mProgram.build({mDevice}) != CL_SUCCESS) {
			std::cout << " Error building: " << mProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevice) << "\n";
			throw std::runtime_error("Error building source code");
		}

		cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(
			cl::Kernel(mProgram, "simple_add"));
		cl::NDRange global(SIZE);
		simple_add(cl::EnqueueArgs(mQueue, global), A_d, B_d, C_d).wait();

		// Retrieve data
		mQueue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

		for(int val : C_h) {
			std::cout << val;
		}
		std::cout << "\n";
	}

	void apply()
	{
		// // Execute the kernel
		// size_t global_work_size = matrix.size();
		// clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
		// clFinish(queue);

		// // Read the result back
		// clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(float) * matrix.size(), matrix.data(), 0, nullptr,
		// nullptr);

		// // Print the matrix
		// for (float val : matrix) {
		//     std::cout << val << " ";
		// }
		// std::cout << std::endl;
	}

	~OpenCLMain()
	{
		// // Cleanup OpenCL resources
		// clReleaseMemObject(buffer);
		// clReleaseKernel(kernel);
		// clReleaseProgram(program);
		// clReleaseCommandQueue(queue);
		// clReleaseContext(context);
	}
};

#endif  // OPENCL_MAIN