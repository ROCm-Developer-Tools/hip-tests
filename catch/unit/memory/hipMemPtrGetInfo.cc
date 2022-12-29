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
 * @addtogroup hipMemPtrGetInfo hipMemPtrGetInfo
 * @{
 * @ingroup MemoryTest
 * `hipMemPtrGetInfo(void* ptr, size_t* size)` -
 * Query memory info.
 * Return snapshot of free memory, and total allocatable memory on the device.
 */

struct MemInfo{
    float a;
    int b;
    void* c;
};

/**
 * Test Description
 * ------------------------
 *  - Allocates specific size of memory for the variables.
 *  - Gets the allocates size and compares it to the initial size.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPtrGetInfo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemPtrGetInfo_Basic") {
  int* iPtr;
  float* fPtr;
  MemInfo* sPtr;
  size_t sSetSize = 1024, sGetSize;
  HIP_CHECK(hipMalloc(&iPtr, sSetSize));
  HIP_CHECK(hipMalloc(&fPtr, sSetSize));
  HIP_CHECK(hipMalloc(&sPtr, sSetSize));
  HIP_CHECK(hipMemPtrGetInfo(iPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);
  HIP_CHECK(hipMemPtrGetInfo(fPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);
  HIP_CHECK(hipMemPtrGetInfo(sPtr, &sGetSize));
  REQUIRE(sGetSize == sSetSize);
}
