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
 * @addtogroup hipModuleOccupancyMaxActiveBlocksPerMultiprocessor hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
 * @{
 * @ingroup OccupancyTest
 * `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
 * int blockSize, size_t dynSharedMemPerBlk)` -
 * Returns occupancy for a device function.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the grid size is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When block size is 0
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_Negative_Parameters") {
  hipModule_t module;
  hipFunction_t function;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  // Get potential blocksize
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, 0));

  // Common negative tests
  MaxActiveBlocksPerMultiprocessorNegative(
      [&function](int* numBlocks, int blockSize, size_t dynSharedMemPerBlk) {
        return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, function, blockSize,
                                                                  dynSharedMemPerBlk);
      },
      blockSize);

  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Test Description
 * ------------------------
 *  - Check if grid size and block size are within valid range using basic kernel functions:
 *    -# When `dynSharedMemPerBlk = 0`
 *      - Expected output: return `hipSuccess`
 *    -# When `dynSharedMemPerBlk = sharedMemPerBlock`
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_Positive_RangeValidation") {
  hipDeviceProp_t devProp;
  hipModule_t module;
  hipFunction_t function;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0") {
    // Get potential blocksize
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, &function](int* numBlocks) {
          return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, function, blockSize,
                                                                    0);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function,
                                                      devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp, &function](int* numBlocks) {
          return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, function, blockSize,
                                                                    devProp.sharedMemPerBlock);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }

  HIP_CHECK(hipModuleUnload(module));
}
