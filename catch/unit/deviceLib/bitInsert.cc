/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip/device_functions.h>

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <random>


// CPU implementation of bitinsert
template <typename T> T bit_insert(T src0, T src1, unsigned int src2, unsigned int src3) {
  unsigned int bits = sizeof(T) * 8;
  T offset = src2 & (bits - 1);
  T width = src3 & (bits - 1);
  T mask = (((T)1) << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__global__ void HIP_kernel(unsigned int* out32, unsigned int* in32_0, unsigned int* in32_1,
                           unsigned int* in32_2, unsigned int* in32_3,
                           unsigned long long int* out64, unsigned long long int* in64_0,
                           unsigned long long int* in64_1, unsigned int* in64_2,
                           unsigned int* in64_3) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  out32[x] = __bitinsert_u32(in32_0[x], in32_1[x], in32_2[x], in32_3[x]);
  out64[x] = __bitinsert_u64(in64_0[x], in64_1[x], in64_2[x], in64_3[x]);
}

TEST_CASE("Unit_bitInsert") {
  using namespace std;

  unsigned int* hostOut32;
  unsigned int* hostSrc032;
  unsigned int* hostSrc132;
  unsigned int* hostSrc232;
  unsigned int* hostSrc332;
  unsigned long long int* hostOut64;
  unsigned long long int* hostSrc064;
  unsigned long long int* hostSrc164;
  unsigned int* hostSrc264;
  unsigned int* hostSrc364;

  unsigned int* deviceOut32;
  unsigned int* deviceSrc032;
  unsigned int* deviceSrc132;
  unsigned int* deviceSrc232;
  unsigned int* deviceSrc332;
  unsigned long long int* deviceOut64;
  unsigned long long int* deviceSrc064;
  unsigned long long int* deviceSrc164;
  unsigned int* deviceSrc264;
  unsigned int* deviceSrc364;

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  INFO("System minor : " << devProp.minor);
  INFO("System major : " << devProp.major);
  INFO("agent prop name : " << devProp.name);

  INFO("hip Device prop succeeded");

  unsigned int wave_size = devProp.warpSize;
  unsigned int num_waves_per_block = 2;
  unsigned int num_threads_per_block = wave_size * num_waves_per_block;
  unsigned int num_blocks = 2;
  unsigned int NUM = num_threads_per_block * num_blocks;

  unsigned i;
  int errors;

  hostOut32 = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostSrc032 = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostSrc132 = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostSrc232 = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostSrc332 = (unsigned int*)malloc(NUM * sizeof(unsigned int));

  hostOut64 = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));
  hostSrc064 = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));
  hostSrc164 = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));
  hostSrc264 = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostSrc364 = (unsigned int*)malloc(NUM * sizeof(unsigned int));

  // initialize the input data
  std::random_device rd;
  std::uniform_int_distribution<uint32_t> uint32_src01_dist;
  std::uniform_int_distribution<uint32_t> uint32_src23_dist(0, 31);
  std::uniform_int_distribution<uint64_t> uint64_src01_dist;
  std::uniform_int_distribution<uint32_t> uint64_src23_dist(0, 63);
  for (i = 0; i < NUM; i++) {
    hostOut32[i] = 0;
    hostSrc032[i] = uint32_src01_dist(rd);
    hostSrc132[i] = uint32_src01_dist(rd);
    hostSrc232[i] = uint32_src23_dist(rd);
    hostSrc232[i] = uint32_src23_dist(rd);
    hostOut64[i] = 0;
    hostSrc064[i] = uint64_src01_dist(rd);
    hostSrc164[i] = uint64_src01_dist(rd);
    hostSrc264[i] = uint64_src23_dist(rd);
    hostSrc264[i] = uint64_src23_dist(rd);
  }

  HIP_CHECK(hipMalloc((void**)&deviceOut32, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc032, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc132, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc232, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc332, NUM * sizeof(unsigned int)));

  HIP_CHECK(hipMalloc((void**)&deviceOut64, NUM * sizeof(unsigned long long int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc064, NUM * sizeof(unsigned long long int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc164, NUM * sizeof(unsigned long long int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc264, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceSrc364, NUM * sizeof(unsigned int)));

  HIP_CHECK(
      hipMemcpy(deviceSrc032, hostSrc032, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(deviceSrc132, hostSrc132, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(deviceSrc232, hostSrc232, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(deviceSrc332, hostSrc332, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpy(deviceSrc064, hostSrc064, NUM * sizeof(unsigned long long int),
                       hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceSrc164, hostSrc164, NUM * sizeof(unsigned long long int),
                       hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(deviceSrc264, hostSrc264, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(deviceSrc364, hostSrc364, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(HIP_kernel, dim3(num_blocks), dim3(num_threads_per_block), 0, 0, deviceOut32,
                     deviceSrc032, deviceSrc132, deviceSrc232, deviceSrc332, deviceOut64,
                     deviceSrc064, deviceSrc164, deviceSrc264, deviceSrc364);
  HIP_CHECK(hipGetLastError());


  HIP_CHECK(hipMemcpy(hostOut32, deviceOut32, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hostOut64, deviceOut64, NUM * sizeof(unsigned long long int),
                       hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostOut32[i] !=
        bit_insert<uint32_t>(hostSrc032[i], hostSrc132[i], hostSrc232[i], hostSrc332[i])) {
      errors++;
      INFO("device: " << hostOut32[i] << " host: "
                      << bit_insert<uint32_t>(hostSrc032[i], hostSrc132[i], hostSrc232[i],
                                              hostSrc332[i])
                      << " " << hostSrc032[i] << " " << hostSrc132[i] << " " << hostSrc232[i] << " "
                      << hostSrc332[i] << "\n");
    }
  }

  for (i = 0; i < NUM; i++) {
    if (hostOut64[i] !=
        bit_insert<uint64_t>(hostSrc064[i], hostSrc164[i], hostSrc264[i], hostSrc364[i])) {
      errors++;
      INFO("device: " << hostOut64[i] << " host: "
                      << bit_insert<uint64_t>(hostSrc064[i], hostSrc164[i], hostSrc264[i],
                                              hostSrc364[i])
                      << " " << hostSrc064[i] << " " << hostSrc164[i] << " " << hostSrc264[i] << " "
                      << hostSrc364[i] << "\n");
    }
  }

  HIP_CHECK(hipFree(deviceOut32));
  HIP_CHECK(hipFree(deviceSrc032));
  HIP_CHECK(hipFree(deviceSrc132));
  HIP_CHECK(hipFree(deviceSrc232));
  HIP_CHECK(hipFree(deviceSrc332));
  HIP_CHECK(hipFree(deviceOut64));
  HIP_CHECK(hipFree(deviceSrc064));
  HIP_CHECK(hipFree(deviceSrc164));
  HIP_CHECK(hipFree(deviceSrc264));
  HIP_CHECK(hipFree(deviceSrc364));

  free(hostOut32);
  free(hostSrc032);
  free(hostSrc132);
  free(hostSrc232);
  free(hostSrc332);
  free(hostOut64);
  free(hostSrc064);
  free(hostSrc164);
  free(hostSrc264);
  free(hostSrc364);

  REQUIRE(errors == 0);
}
