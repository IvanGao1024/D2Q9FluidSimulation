#ifndef OPENCL_MAIN
#define OPENCL_MAIN

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <cassert>
#include <vector>
#include <iostream>
#include <memory>

class OpenCLMain
{
private:
	struct MachineProfile {
		std::string mPlatformName;
		std::string mDeviceName;
	};

private:
	static inline MachineProfile UserMachineProfile;

	static inline cl::Platform         mPlatform;
	static inline cl::Device           mDevice;
	static inline cl::Context          mContext;
	static inline cl::CommandQueue     mQueue;
	static inline cl::Program::Sources mSources;
	static inline cl::Program          mProgram;

	static inline cl::NDRange local;
	static inline cl::NDRange global;

	static inline std::unique_ptr<cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>> pArrayAdditionKernel;
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

		// Handel context
		mContext = cl::Context({mDevice});

		// Handel queue
		mQueue = cl::CommandQueue(mContext, mDevice);

		// Handel kernel
		std::string kernelCode = R"(
            void kernel arrayAdditionKernel(global const int* A, global const int* B, global int* C){
                C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];
            }
        )";

		mSources.push_back({kernelCode.c_str(), kernelCode.length()});
		mProgram = cl::Program(mContext, mSources);

		if(mProgram.build({mDevice}) != CL_SUCCESS) {
			std::cout << " Error building: " << mProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevice) << "\n";
			throw std::runtime_error("Error building source code");
		}
		

		pArrayAdditionKernel = std::make_unique<cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>>(
        	cl::Kernel(mProgram, "arrayAdditionKernel")
    	);
		local = cl::NDRange(256);
	}

public:
	static OpenCLMain& instance() {
        static OpenCLMain instance; // Guaranteed to be destroyed and instantiated on first use
        return instance;
	}

	/**
	 * @brief Add two array using OpenCL, require A_h and B_h has the same size.
	 * 
	 * @param A_h 
	 * @param B_h 
	 * @return std::vector<unsigned int> C_h stores the resulting array.
	 */
	static void arrayAddition(const std::vector<unsigned int> A_h, const std::vector<unsigned int> B_h, std::vector<unsigned int> C_h)
	{	
		assert(A_h.size() == B_h.size());
		assert(B_h.size() == C_h.size());
		assert(A_h.size() == C_h.size());

		const unsigned int SIZE = A_h.size();
		global = cl::NDRange(SIZE);

		cl::Buffer A_d(mContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * SIZE);
		cl::Buffer B_d(mContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * SIZE);
		cl::Buffer C_d(mContext, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * SIZE);

		// Example cont'd
		mQueue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(unsigned int) * SIZE, A_h.data());
		mQueue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(unsigned int) * SIZE, B_h.data());
		
		(*pArrayAdditionKernel)(cl::EnqueueArgs(mQueue, global, local), A_d, B_d, C_d).wait();
        
		// arrayAdditionKernel(cl::EnqueueArgs(mQueue, global, local), A_d, B_d, C_d).wait();
        
		// Retrieve data
		mQueue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h.data());

		// for (size_t i = 0; i < 20; i++)
		// {
		// 	std::cout << C_h[i];
		// }
		// std::cout << "\n";
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
	
	// Delete copy/move constructors and assignment operators
	OpenCLMain(OpenCLMain const&) = delete;
	void operator=(OpenCLMain const&) = delete;
	OpenCLMain(OpenCLMain&&) = delete;
	void operator=(OpenCLMain&&) = delete;
};

#endif  // OPENCL_MAIN