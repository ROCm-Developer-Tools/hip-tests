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
 * @addtogroup hipModuleOccupancyMaxPotentialBlockSizeWithFlags hipModuleOccupancyMaxPotentialBlockSizeWithFlags
 * @{
 * @ingroup OccupancyTest
 * `hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
 * hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int  flags)` -
 * Determine the grid and block sizes to achieves maximum occupancy for a kernel.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the grid size is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When output pointer to the block size is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When flag is invalid, because only default flag is supported
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipModuleOccupancyMaxPotentialBlockSizeWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Negative_Parameters") {
  hipModule_t module;
  hipFunction_t function;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  // Common negative tests
  MaxPotentialBlockSizeNegative([&function](int* gridSize, int* blockSize) {
    return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, function, 0, 0,
                                                            hipOccupancyDefault);
  });

#if HT_NVIDIA // EXSWHTEC-219
  SECTION("Flag is invalid") {
    // Only default flag is supported
    HIP_CHECK_ERROR(
        hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize, &blockSize, function, 0, 0, 2),
        hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipModuleUnload(module));
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
 *  - unit/occupancy/hipModuleOccupancyMaxPotentialBlockSizeWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Positive_RangeValidation") {
  hipDeviceProp_t devProp;
  hipModule_t module;
  hipFunction_t function;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [&function](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, function, 0,
                                                                  0, hipOccupancyDefault);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [&function, devProp](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
              gridSize, blockSize, function, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock,
              hipOccupancyDefault);
        },
        devProp.maxThreadsPerBlock);
  }

  HIP_CHECK(hipModuleUnload(module));
}
