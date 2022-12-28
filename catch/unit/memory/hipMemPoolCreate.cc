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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <hip_test_common.hh>

#include <limits>

/**
 * @addtogroup hipMemPoolCreate hipMemPoolCreate
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props)` -
 * Creates a HIP memory pool and returns the handle in mem pool.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemPoolApi_Basic
 *  - @ref Unit_hipMemPoolApi_BasicAlloc
 */

constexpr hipMemPoolProps kPoolPropsValid = {
    hipMemAllocationTypePinned, hipMemHandleTypeNone, {hipMemLocationTypeDevice, 0}, nullptr, {0}};

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the memory pool is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When properties are `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPoolCreate.cc
 * Test requirements
 * ------------------------
 *  - Runtime supports Memory Pools
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemPoolCreate_Negative_Parameter") {
  HIP_CHECK(hipSetDevice(0));
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  hipMemPool_t mem_pool = nullptr;
  size_t max_size = std::numeric_limits<size_t>::max();

  SECTION("mem_pool ptr is nullptr") {
    REQUIRE(hipMemPoolCreate(nullptr, &kPoolPropsValid) != hipSuccess);
  }

  SECTION("properties is null") { REQUIRE(hipMemPoolCreate(&mem_pool, nullptr) != hipSuccess); }
}
