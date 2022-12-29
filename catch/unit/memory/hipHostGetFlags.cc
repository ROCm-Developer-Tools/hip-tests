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
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <thread>
#include <vector>

/**
 * @addtogroup hipHostGetFlags hipHostGetFlags
 * @{
 * @ingroup MemoryTest
 * `hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr)` -
 *  Return flags associated with host pointer.
 */

std::vector<unsigned int> FlagPart1Vec{hipHostMallocDefault,
                                       hipHostMallocDefault | hipHostMallocPortable,
                                       hipHostMallocDefault | hipHostMallocMapped,
                                       hipHostMallocDefault | hipHostMallocWriteCombined,
                                       hipHostMallocPortable,
                                       hipHostMallocPortable | hipHostMallocMapped,
                                       hipHostMallocPortable | hipHostMallocWriteCombined,
                                       hipHostMallocMapped,
                                       hipHostMallocMapped | hipHostMallocWriteCombined,
                                       hipHostMallocWriteCombined};
#if HT_AMD
// For cases where flags from FlagPart1Vec are not used,
// hipHostMallocDefault is the default on AMD
// and hipHostMallocMapped on Nvidia
std::vector<unsigned int> FlagPart2Vec{0x0,
                                       hipHostMallocNumaUser,
                                       hipHostMallocNumaUser | hipHostMallocCoherent,
                                       hipHostMallocNumaUser | hipHostMallocNonCoherent,
                                       hipHostMallocCoherent,
                                       hipHostMallocNonCoherent};
#else
std::vector<unsigned int> FlagPart2Vec{0x0};
#endif

static constexpr auto LEN{1024 * 1024};

inline void checkFlags(unsigned int expected, unsigned int obtained) {
  // Account for cases where flags from FlagPart1Vec do not include hipHostMallocMapped,
  // on Nvidia devices it is added by default
#if HT_NVIDIA
  expected = expected | hipHostMallocMapped;
#endif
  REQUIRE(expected == obtained);
}

/**
 * Test Description
 * ------------------------
 *  - Validate that flags set when allocating memory are returned.
 *  - Allocate memory, and get flags.
 *  - Check that retreived flags are equal to the ones that are set.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostGetFlags.cc
 * Test requirements
 * ------------------------
 *  - Device supports mapping of host memory
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipHostGetFlags_flagCombos") {

  constexpr auto SIZE{LEN * sizeof(int)};
  int* A_h{nullptr};

  const unsigned int FlagPart1 = GENERATE(from_range(FlagPart1Vec.begin(), FlagPart1Vec.end()));
  const unsigned int FlagPart2 = GENERATE(from_range(FlagPart2Vec.begin(), FlagPart2Vec.end()));

  unsigned int FlagComp = FlagPart1 | FlagPart2;

  hipDeviceProp_t prop;
  int device{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  // Skip test if device does not support the property canMapHostMemory
  if (prop.canMapHostMemory != 1) {
    HipTest::HIP_SKIP_TEST("Device Property canMapHostMemory is not set");
    return;
  } else {
    // Allocate using the generated flags combos
    INFO("Flag passed when allocating: 0x" << std::hex << FlagComp << "\n");
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), SIZE, FlagComp));
    unsigned int flagA{};

    // get the flags from allocations and check if they are the same as the one set
    HIP_CHECK(hipHostGetFlags(&flagA, A_h));

    checkFlags(FlagComp, flagA);
    HIP_CHECK(hipHostFree(A_h));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validate that flags set when allocating memory in a separate thread,
 *    are returned.
 *  - Launch thread that allocates memory.
 *  - Join thread.
 *  - Get flags and check that they are equal to the ones that were set.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostGetFlags.cc
 * Test requirements
 * ------------------------
 *  - Device supports mapping of host memory
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipHostGetFlags_DifferentThreads") {
  constexpr auto SIZE{LEN * sizeof(int)};
  int* A_h{nullptr};

  const unsigned int FlagPart1 = GENERATE(from_range(FlagPart1Vec.begin(), FlagPart1Vec.end()));
  const unsigned int FlagPart2 = GENERATE(from_range(FlagPart2Vec.begin(), FlagPart2Vec.end()));


  unsigned int FlagComp = FlagPart1 | FlagPart2;

  hipDeviceProp_t prop;
  int device{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  if (prop.canMapHostMemory != 1) {
    HipTest::HIP_SKIP_TEST("Device Property canMapHostMemory is not set");
    return;
  } else {
    // Make sure we allocate before trying to get the flags
    std::thread malloc_thread(
        [&]() { HIP_CHECK_THREAD(hipHostMalloc(reinterpret_cast<void**>(&A_h), SIZE, FlagComp)); });
    malloc_thread.join();
    HIP_CHECK_THREAD_FINALIZE();
    unsigned int flagA{};
    HIP_CHECK(hipHostGetFlags(&flagA, A_h));

    checkFlags(FlagComp, flagA);

    HIP_CHECK(hipHostFree(A_h));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the output flag pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When passing device pointer instead of a host pointer
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When passing device pointer instead of a host pointer,
 *       using `hipHostGetDevicePointer`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipHostGetFlags.cc
 * Test requirements
 * ------------------------
 *  - Device supports mapping of host memory
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipHostGetFlags_InvalidArgs") {
  constexpr auto SIZE{LEN * sizeof(int)};
  int* A_h{nullptr};

  hipDeviceProp_t prop;
  int device{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  // Skip test if device does not support the property canMapHostMemory
  if (prop.canMapHostMemory != 1) {
    HipTest::HIP_SKIP_TEST("Device Property canMapHostMemory is not set");
    return;
  } else {
    SECTION("Invalid flag ptr being passed to hipHostGetFlags") {
      // Use default flag
      unsigned int FlagComp = 0x0;

      // Allocate using the generated flags combos
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), SIZE, FlagComp));

      // use a nullptr to return flags to
      unsigned int* flagA = nullptr;

      // get the flags from allocations and check if they are the same as the one set
      HIP_CHECK_ERROR(hipHostGetFlags(flagA, A_h), hipErrorInvalidValue);

      HIP_CHECK(hipHostFree(A_h));
    }

    SECTION("Device ptr allocated with hipMalloc passed to hipHostGetFlags") {
      unsigned int FlagComp = 0x4;

      // Allocate memory on device
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_h), SIZE));

      unsigned int flagA{};

      // get the flags from allocations and check if they are the same as the one set
      HIP_CHECK_ERROR(hipHostGetFlags(&flagA, A_h), hipErrorInvalidValue);
      INFO("Flag passed when allocating: " << std::hex << FlagComp << " Returned flag: " << std::hex
                                           << flagA << "\n");

      HIP_CHECK(hipFree(A_h));
    }

    SECTION("Ptr from hipHostGetDevicePointer passed to hipHostGetFlags") {
      unsigned int FlagComp = 0x4;

      int* A_d{nullptr};
      // Allocate memory on device
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), SIZE, FlagComp));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));

      unsigned int flagA;

      // get the flags from allocations and check if they are the same as the one set
      HIP_CHECK(hipHostGetFlags(&flagA, A_d));
      INFO("Flag passed when allocating: " << std::hex << FlagComp << " Returned flag: " << std::hex
                                           << flagA << "\n");
#if HT_NVIDIA
      // on Nvidia adjust for cudaHostAllocMapped being set by default
      FlagComp = FlagComp | hipHostMallocMapped;
#endif
      REQUIRE(flagA == FlagComp);
      HIP_CHECK(hipHostFree(A_h));
    }
  }
}
