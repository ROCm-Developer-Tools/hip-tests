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

#include <chrono>
#include <hip_test_common.hh>

/**
 * @addtogroup hipStreamCreateWithFlags hipStreamCreateWithFlags
 * @{
 * @ingroup StreamTest
 * `hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags)` -
 * Create an asynchronous stream with flags.
 */

namespace hipStreamCreateWithFlagsTests {

/**
 * Test Description
 * ------------------------
 *  - Verifies handling of invalid arguments:
 *    -# When output pointer to the stream is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamCreateWithFlags_Negative_NullStream") {
  HIP_CHECK_ERROR(hipStreamCreateWithFlags(nullptr, hipStreamDefault), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Creates stream with invalid flag.
 *  - Valid flags are 0x0 and 0x1.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamCreateWithFlags_Negative_InvalidFlag") {
  hipStream_t stream{};
  unsigned int flag = 0xFF;
  REQUIRE(flag != hipStreamDefault);
  REQUIRE(flag != hipStreamNonBlocking);
  HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, flag), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Creates streams with valid flags.
 *  - Checks that they are created as expected.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamCreateWithFlags_Default") {
  const unsigned int flagUnderTest = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&stream, flagUnderTest));

  unsigned int flag{};
  HIP_CHECK(hipStreamGetFlags(stream, &flag));
  REQUIRE(flag == flagUnderTest);

  int priority{};
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  // zero is considered default priority
  REQUIRE(priority == 0);

  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */
/**
 * Test Description
 * ------------------------
 *  - Test how stream set as default interacts with created streams.
 *    -# When null stream is set as default and stream is created as default
 *      - Created stream is blocking the null stream and vice versa
 *    -# When null stream is set as default and stream is created as non blocking
 *      - Created stream is not blocking the null stream and vice versa
 *    -# When stream per thread is set as default and stream is created with any flag
 *      - Created stream is not blocking the default stream and vice versa
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamCreateWithFlags_DefaultStreamInteraction") {
  const hipStream_t defaultStream = GENERATE(static_cast<hipStream_t>(nullptr), hipStreamPerThread);
  const unsigned int flagUnderTest = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  const hipError_t expectedError = (flagUnderTest == hipStreamDefault) && (defaultStream == nullptr)
      ? hipErrorNotReady
      : hipSuccess;
  CAPTURE(defaultStream, flagUnderTest, expectedError, hipGetErrorString(expectedError));

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&stream, flagUnderTest));

  constexpr auto delay = std::chrono::milliseconds(500);

  SECTION("default stream waiting for created stream") {
    HipTest::runKernelForDuration(delay, stream);
    REQUIRE(hipStreamQuery(defaultStream) == expectedError);
  }
  SECTION("created stream waiting for default stream") {
    HipTest::runKernelForDuration(delay, defaultStream);
    REQUIRE(hipStreamQuery(stream) == expectedError);
  }

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

}  // namespace hipStreamCreateWithFlagsTests
