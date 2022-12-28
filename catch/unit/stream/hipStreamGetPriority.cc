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

/**
 * @addtogroup hipStreamGetPriority hipStreamGetPriority
 * @{
 * @ingroup StreamTest
 * `hipStreamGetPriority(hipStream_t stream, int* priority)` -
 * Query the priority of a stream.
 */

/**
 * Test Description
 * ------------------------
 *  - Checks different valid scenarios:
 *    -# When stream is `nullptr`
 *      - Expected output: valid priority
 *    -# When default priority stream is created
 *      - Expected output: valid priority
 *    -# When high priority stream is created
 *      - Expected output: valid priority
 *    -# When stream priority is higher than avaliable
 *      - Expected output: clamped priority to the highest valid one
 *    -# When low priority stream is created
 *      - Expected output: valid priority 
 *    -# When stream priority is lower than available
 *      - Expected output: clamped priority to the lowest valid one
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetPriority_happy") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  int priority = 0;
  SECTION("Null Stream") {
    HIP_CHECK(hipStreamGetPriority(nullptr, &priority));
    // valid priority
    REQUIRE(priority_low >= priority);
    REQUIRE(priority >= priority_high);
  }
  SECTION("Created Stream") {
    SECTION("Default Priority") {
      HIP_CHECK(hipStreamCreate(&stream));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      // valid priority
      // Lower the value higher the priority, higher the value lower the priority
      REQUIRE(priority_low >= priority);
      REQUIRE(priority >= priority_high);
    }
    SECTION("High Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority == priority_high);
    }
    SECTION("Higher Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_high - 1));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority == priority_high);
    }
    SECTION("Low Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority_low == priority);
    }
    SECTION("Lower Priority") {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_low + 1));
      HIP_CHECK(hipStreamGetPriority(stream, &priority));
      REQUIRE(priority_low == priority);
    }
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies the case when both stream and priority pointers are `nullptr`
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetPriority_nullptr_nullptr") {
  auto res = hipStreamGetPriority(nullptr,nullptr);
  REQUIRE(res == hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Verifies the case when priority pointer is `nullptr`
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetPriority_stream_nullptr") {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  auto res = hipStreamGetPriority(stream,nullptr);
  REQUIRE(res == hipErrorInvalidValue);

  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies the case when stream pointer is `nullptr`
 *    - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetPriority_nullptr_priority") {
  int priority = -1;
  HIP_CHECK(hipStreamGetPriority(nullptr,&priority));
}

/**
 * Test Description
 * ------------------------
 *  - Both stream and priority pointers are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetPriority_stream_priority") {
  int priority = -1;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipStreamGetPriority(stream,&priority));

  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Create stream with CU mask and check priority is returned as expected.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamGetPriority_StreamsWithCUMask") {
  hipStream_t stream{};
  int priority = 0;
  int priority_normal = 0;
  int priority_low = 0;
  int priority_high = 0;
  // Test is to get the Stream Priority Range
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  priority_normal = (priority_low + priority_high) / 2;
  // Check if priorities are indeed supported
  REQUIRE_FALSE(priority_low == priority_high);
  // Creating a stream with hipExtStreamCreateWithCUMask and checking
  // priority.
  const uint32_t cuMask = 0xffffffff;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, 1, &cuMask));
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE_FALSE(priority_normal != priority);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
