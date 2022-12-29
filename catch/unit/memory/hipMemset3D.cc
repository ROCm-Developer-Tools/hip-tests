/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @addtogroup hipMemset3D hipMemset3D
 * @{
 * @ingroup MemoryTest
 * `hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent )` -
 * Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
 */

/**
 * Test Description
 * ------------------------
 *  - Allocates 3D memory chunk.
 *  - Sets memory to the unique value.
 *  - Performs copy and validates results.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemset3D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemset3D_BasicFunctional") {
  constexpr int memsetval = 0x22;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW * numH * depth;
  char *A_h;

  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
      A_h[i] = 1;
  }
  HIP_CHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width , numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif
  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
      if (A_h[i] != memsetval) {
        INFO("Memset3D mismatch at index:" << i << " computed:"
                                      << A_h[i] << " memsetval:" << memsetval);
        REQUIRE(false);
      }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}

/**
 * End doxygen group hipMemset3D.
 * @}
 */

/**
 * @addtogroup hipMemset3DAsync hipMemset3DAsync
 * @{
 * @ingroup MemoryTest
 * `hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value,
 * hipExtent extent ,hipStream_t stream __dparm(0))` -
 * Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemsetFunctional_ZeroValue_3D
 *  - @ref Unit_hipMemsetFunctional_SmallSize_3D
 *  - @ref Unit_hipMemsetFunctional_ZeroSize_3D
 *  - @ref Unit_hipMemsetFunctional_PartialSet_3D
 */

/**
 * Test Description
 * ------------------------
 *  - Sets 3D allocated pitch memory, asynchronously.
 *  - Performs copy and compares the results.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemset3D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemset3DAsync_BasicFunctional") {
  constexpr int memsetval = 0x22;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW * numH * depth;
  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;
  char *A_h;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
      A_h[i] = 1;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width , numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif
  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
      if (A_h[i] != memsetval) {
        INFO("Memset3DAsync mismatch at index:" << i << " computed:"
                                      << A_h[i] << " memsetval:" << memsetval);
        REQUIRE(false);
      }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}
