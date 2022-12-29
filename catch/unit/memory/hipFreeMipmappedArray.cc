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


#include <hip_test_common.hh>
#include "hipArrayCommon.hh"
#include "utils.hh"

/**
 * @addtogroup hipFreeMipmappedArray hipFreeMipmappedArray
 * @{
 * @ingroup MemoryTest
 * `hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray)` -
 * Frees a mipmapped array on the device.
 */

/**
 * Test Description
 * ------------------------
 *  - Frees an array when the device is busy.
 *  - Uses query on null stream for synchronization status.
 * Test source
 * ------------------------
 *  - unit/memory/hipFreeMipmappedArray.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS)
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipFreeMipmappedArrayImplicitSyncArray", "", char, float) {
#ifdef _WIN32
  hipMipmappedArray_t arrayPtr{};
  hipExtent extent{};
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

#if HT_AMD
  const unsigned int flags = hipArrayDefault;
#else
  const unsigned int flags = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
#endif

  extent.width = GENERATE(64, 256, 1024);
  extent.height = GENERATE(64, 256, 1024);
  extent.depth = GENERATE(0, 64, 256, 1024);

  const unsigned int numLevels = GENERATE(1, 5, 7);

  HIP_CHECK(hipMallocMipmappedArray(&arrayPtr, &desc, extent, numLevels, flags));

  LaunchDelayKernel(std::chrono::milliseconds{50}, nullptr);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(hipFreeMipmappedArray(arrayPtr));
  HIP_CHECK(hipStreamQuery(nullptr));
#else
  HipTest::HIP_SKIP_TEST("Mipmaps are Supported only on windows, skipping the test.");
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when mipmapped array handle is `nullptr`
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipFreeMipmappedArray.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipFreeMipmappedArray_Negative_Nullptr") {
#ifdef _WIN32
  HIP_CHECK_ERROR(hipFreeMipmappedArray(nullptr), hipErrorInvalidValue);
#else
  HipTest::HIP_SKIP_TEST("Mipmaps are Supported only on windows, skipping the test.");
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when free is called twice
 *    - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/memory/hipFreeMipmappedArray.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipFreeMipmappedArray_Negative_DoubleFree") {
#ifdef _WIN32
  hipMipmappedArray_t arrayPtr{};
  hipExtent extent{};
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

#if HT_AMD
  const unsigned int flags = hipArrayDefault;
#else
  const unsigned int flags = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
#endif

  extent.width = GENERATE(64, 512, 1024);
  extent.height = GENERATE(64, 512, 1024);
  extent.depth = GENERATE(0, 64, 512, 1024);

  const unsigned int numLevels = GENERATE(1, 5, 7);

  HIP_CHECK(hipMallocMipmappedArray(&arrayPtr, &desc, extent, numLevels, flags));

  HIP_CHECK(hipFreeMipmappedArray(arrayPtr));
  HIP_CHECK_ERROR(hipFreeMipmappedArray(arrayPtr), hipErrorContextIsDestroyed);
#else
  HipTest::HIP_SKIP_TEST("Mipmaps are Supported only on windows, skipping the test.");
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Frees multiple arrays from multiple threads.
 * Test source
 * ------------------------
 *  - unit/memory/hipFreeMipmappedArray.cc
 * Test requirements
 * ------------------------
 *  - Multi-threaded device
 *  - Host specific (WINDOWS)
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipFreeMipmappedArrayMultiTArray", "", char, int) {
#ifdef _WIN32
  constexpr size_t numAllocs = 10;
  std::vector<std::thread> threads;
  std::vector<hipMipmappedArray_t> ptrs(numAllocs);
  hipExtent extent{};
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  const unsigned int numLevels = GENERATE(1, 5, 7);

#if HT_AMD
  const unsigned int flags = hipArrayDefault;
#else
  const unsigned int flags = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
#endif

  extent.width = GENERATE(64, 256, 1024);
  extent.height = GENERATE(64, 256, 1024);
  extent.depth = GENERATE(0, 64, 256, 1024);

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipMallocMipmappedArray(&ptr, &desc, extent, numLevels, flags));
  }

  for (auto ptr : ptrs) {
    threads.emplace_back(([ptr] {
      HIP_CHECK_THREAD(hipFreeMipmappedArray(ptr));
      HIP_CHECK_THREAD(hipStreamQuery(nullptr));
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
  
  HIP_CHECK_THREAD_FINALIZE();
#else
  HipTest::HIP_SKIP_TEST("Mipmaps are Supported only on windows, skipping the test.");
#endif
}
