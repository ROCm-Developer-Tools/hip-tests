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

#include "occupancy_common.hh"

/**
 * @addtogroup hipOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
 * @{
 * @ingroup OccupancyTest
 * `hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
 * const void* f, size_t dynSharedMemPerBlk, int blockSizeLimit)` -
 * Determine the grid and block sizes to achieves maximum occupancy for a kernel.
 */

static __global__ void f1(float* a) { *a = 1.0; }

template <typename T> static __global__ void f2(T* a) { *a = 1; }

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the grid size is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When output pointer to the block size is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pointer to the function is `nullptr`
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidDeviceFunction`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipOccupancyMaxPotentialBlockSize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Negative_Parameters") {
  // Common negative tests
  MaxPotentialBlockSizeNegative([](int* gridSize, int* blockSize) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f1, 0, 0);
  });

#if HT_AMD
#if 0 // EXSWHTEC-219
  SECTION("Kernel function is NULL") {
    int blockSize = 0;
    int gridSize = 0;
    // nvcc doesnt support kernelfunc(NULL) for api
    HIP_CHECK_ERROR(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, NULL, 0, 0),
                    hipErrorInvalidDeviceFunction);
  }
#endif
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Check if grid size and block size are within valid range using basic kernel functions:
 *    -# When `dynSharedMemPerBlk = 0, blockSizeLimit = 0`
 *      - Expected output: return `hipSuccess`
 *    -# When `dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock`
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipOccupancyMaxPotentialBlockSize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Positive_RangeValidation") {
  hipDeviceProp_t devProp;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f1, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [devProp](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize(
              gridSize, blockSize, f1, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Check is number of blocks is greater than 0 when API is invoked with a template:
 *    -# When `dynSharedMemPerBlk = 0, blockSizeLimit = 0`
 *      - Expected output: return `hipSuccess`
 *    -# When `dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock`
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipOccupancyMaxPotentialBlockSize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Positive_TemplateInvocation") {
  hipDeviceProp_t devProp;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(gridSize, blockSize, f2, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [devProp](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(
              gridSize, blockSize, f2, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }
}
