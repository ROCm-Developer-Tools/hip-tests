/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include"hip_runtime.h"
#include<iostream>
#include<time.h>

#define check(msg, status) \
if(status != hipSuccess){ \
	printf("%s failed.\n",#msg); \
	exit(1); \
}

#define LEN 1024*1024
#define SIZE LEN * sizeof(float)
#define ITER 10000

__global__ void One(hipLaunchParm lp, float* Ad){
}

int main(){

	hipError_t err;
	float *A, *Ad;

	A = new float[LEN];

	for(int i=0;i<LEN;i++){
		A[i] = 1.0f;
	}

	hipStream_t stream;
	err = hipStreamCreate(&stream);
	check("Creating stream",err);

	err = hipMalloc(&Ad, SIZE);
	check("Allocating Ad memory on device", err);

	err = hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
	check("Doing memory copy from A to Ad", err);

	float mS = 0;
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	hipEventRecord(start);
	hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, 0, Ad);
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"First Kernel Launch: \t\t"<<mS*1000<<" uS"<<std::endl;
	
	hipEventRecord(start);
	hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, 0, Ad);
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"Second Kernel Launch: \t\t"<<mS*1000<<" uS"<<std::endl;
	
	hipEventRecord(start);
	for(int i=0;i<ITER;i++){
		hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, 0, Ad);
	}
	hipDeviceSynchronize();
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"NULL Stream Sync dispatch wait: \t"<<mS*1000/ITER<<" uS"<<std::endl;
	hipDeviceSynchronize();

	hipEventRecord(start);
	for(int i=0;i<ITER;i++){
		hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, 0, Ad);
	}
	hipEventRecord(stop);
	hipDeviceSynchronize();
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"NULL Stream Async dispatch wait: \t"<<mS*1000/ITER<<" uS"<<std::endl;
	hipDeviceSynchronize();

	hipEventRecord(start);
	for(int i=0;i<ITER;i++){
		hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, stream, Ad);
		hipDeviceSynchronize();
	}
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"Stream Sync dispatch wait: \t\t"<<mS*1000/ITER<<" uS"<<std::endl;
	hipDeviceSynchronize();
	hipEventRecord(start);
	for(int i=0;i<ITER;i++){
		hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, stream, Ad);
	}
	hipDeviceSynchronize();
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"Stream Async dispatch wait: \t\t"<<mS*1000/ITER<<" uS"<<std::endl;
	hipDeviceSynchronize();

	hipEventRecord(start);
	for(int i=0;i<ITER;i++){
		hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, 0, Ad);
	}
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"NULL Stream Dispatch No Wait: \t\t"<<mS*1000/ITER<<" uS"<<std::endl;
	hipDeviceSynchronize();

	hipEventRecord(start);
	for(int i=0;i<ITER;i++){
		hipLaunchKernel(HIP_KERNEL_NAME(One), dim3(LEN/512), dim3(512), 0, stream, Ad);
	}
	hipEventRecord(stop);
	hipEventElapsedTime(&mS, start, stop);
	std::cout<<"Stream Dispatch No Wait: \t\t"<<mS*1000/ITER<<" uS"<<std::endl;
	hipDeviceSynchronize();
}
