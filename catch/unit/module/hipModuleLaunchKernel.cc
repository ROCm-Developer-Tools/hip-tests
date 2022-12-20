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
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipModuleLaunchKernel hipModuleLaunchKernel
 * @{
 * @ingroup ModuleTest
 * `hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
 * unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
 * unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra)` -
 * Launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra.
 */

static hipError_t hipModuleLaunchKernelWrapper(hipFunction_t f, uint32_t gridX, uint32_t gridY,
                                               uint32_t gridZ, uint32_t blockX, uint32_t blockY,
                                               uint32_t blockZ, size_t sharedMemBytes,
                                               hipStream_t hStream, void** kernelParams,
                                               void** extra, hipEvent_t, hipEvent_t, uint32_t) {
  return hipModuleLaunchKernel(f, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
                               hStream, kernelParams, extra);
}

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
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Basic") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelPositiveBasic<hipModuleLaunchKernelWrapper>();
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
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelPositiveParameters<hipModuleLaunchKernelWrapper>();
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
 *  - unit/module/hipModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleLaunchKernel_Negative_Parameters") {
  HIP_CHECK(hipFree(nullptr));
  ModuleLaunchKernelNegativeParameters<hipModuleLaunchKernelWrapper>();
}
