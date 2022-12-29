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

#include "vulkan_test.hh"

/**
 * @addtogroup hipDestroyExternalSemaphore hipDestroyExternalSemaphore
 * @{
 * @ingroup MemoryTest
 * `hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem)` -
 * Destroys an external semaphore object and releases any references to the underlying resource.
 * Any outstanding signals or waits must have completed before the semaphore is destroyed.
 */

constexpr bool enable_validation = false;

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the external semaphore is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When semaphore has already been destroyed:
 *      - Platform specific (AMD)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/vulkan_interop/hipDestroyExternalSemaphore.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDestroyExternalSemaphore_Vulkan_Negative_Parameters") {
  SECTION("extSem == nullptr") {
    HIP_CHECK_ERROR(hipDestroyExternalSemaphore(nullptr), hipErrorInvalidValue);
  }

// Segfaults in CUDA
#if HT_AMD
  SECTION("Double free") {
    VulkanTest vkt(enable_validation);
    const auto ext_semaphore = ImportBinarySemaphore(vkt);
    HIP_CHECK(hipDestroyExternalSemaphore(ext_semaphore));
    HIP_CHECK_ERROR(hipDestroyExternalSemaphore(ext_semaphore), hipErrorInvalidValue);
  }
#endif
}