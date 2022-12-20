/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip_module_launch_kernel_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_ext.h>

/**
 * @addtogroup hipExtModuleLaunchKernel hipExtModuleLaunchKernel
 * @{
 * @ingroup ModuleTest
 * `hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, 
 * uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
 * uint32_t localWorkSizeX, uint32_t localWorkSizeY,
 * uint32_t localWorkSizeZ, size_t sharedMemBytes,
 * hipStream_t hStream, void** kernelParams, void** extra,
 * hipEvent_t startEvent = nullptr,
 * hipEvent_t stopEvent = nullptr, uint32_t flags = 0)` -
 * Launches kernel with parameters and shared memory on stream with arguments passed
 * to kernel params or extra arguments.
 */

/**
 * Test Description
 * ------------------------
 *  - Launch kernels in different basic scenarios:
 *    -# When kernel is launched with no arguments
 *      - Expected output: return `hipSuccess`
 *    -# When kernel is launched with arguments using `kernelParams`
 *      - Expected output: return `hipSuccess`
 *    -# When kernel is launched with arguments using `extra`
 *      - Expected output: return `hipSuccess`
 *    -# When kernel is launched as timed and with events
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/module/hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Basic") {
  ModuleLaunchKernelPositiveBasic<hipExtModuleLaunchKernel>();

  SECTION("Timed kernel launch with events") {
    hipEvent_t start_event = nullptr, stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "Delay");
    int clock_rate = 0;
    HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
    uint32_t interval = 100;
    uint32_t ticks_per_second = clock_rate;
    void* kernel_params[2] = {&interval, &ticks_per_second};
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_params, nullptr,
                                       start_event, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    auto elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start_event, stop_event));
    REQUIRE(static_cast<uint32_t>(elapsed) >= interval);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernel with different valid arguments:
 *    -# When gridDimX == maxGridDimX
 *      - Expected output: return `hipSuccess`
 *    -# When gridDimY == maxGridDimY
 *      - Expected output: return `hipSuccess`
 *    -# When gridDimZ == maxGridDimZ
 *      - Expected output: return `hipSuccess`
 *    -# When blockDimX == maxBlockDimX
 *      - Expected output: return `hipSuccess`
 *    -# When blockDimY == maxBlockDimY
 *      - Expected output: return `hipSuccess`
 *    -# When blockDimZ == maxBlockDimZ
 *      - Expected output: return `hipSuccess`
 *    -# When only start event is passed
 *      - Expected output: return `hipSucccess`
 *    -# When only stop event is passed
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/module/hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Parameters") {
  ModuleLaunchKernelPositiveParameters<hipExtModuleLaunchKernel>();

  SECTION("Pass only start event") {
    hipEvent_t start_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       start_event, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(start_event));
  }

  SECTION("Pass only stop event") {
    hipEvent_t stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       nullptr, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(stop_event));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the kernel function is `nullptr`
 *      - Expected output: return `hipErrorInvalidResourceHandle`
 *    -# When gridDimX == 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When gridDimY == 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When gridDimZ == 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimX == 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimY == 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimZ == 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When gridDimX > maxGridDimX
 *      - Disabled on AMD because is returns `hipSuccess`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When gridDimY > maxGridDimY
 *      - Disabled on AMD because is returns `hipSuccess`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When gridDimZ > maxGridDimZ
 *      - Disabled on AMD because is returns `hipSuccess`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimX > maxBlockDimX
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimY > maxBlockDimY
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimZ > maxBlockDimZ
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When blockDimX * blockDimY * blockDimZ > MaxThreadsPerBlock
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When sharedMemBytes > max shared memory per block
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When stream is not valid, e.g., it is destroyed before kernel launch
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When passing kernelArgs and extra simultaneously
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When extra is not valid
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/module/hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_Negative_Parameters") {
  ModuleLaunchKernelNegativeParameters<hipExtModuleLaunchKernel>();
}
