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
 * @addtogroup hipImportExternalMemory hipImportExternalMemory
 * @{
 * @ingroup MemoryTest
 * `hipImportExternalMemory(hipExternalMemory_t* extMem_out,
 * const hipExternalMemoryHandleDesc* memHandleDesc)` -
 * Imports an external memory object.
 */

constexpr bool enable_validation = false;

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When external memory pointer is `nullptr`
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memory handle descriptor is `nullptr`
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue
 *    -# When size within memory handle descriptor is 0
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When flags within memory handle descriptor are not valid
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When type within memory handle descriptor is not valid (-1)
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When handle within memory handle descriptor is `nullptr`
 *      - Host specific (WINDOWS)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/vulkan_interop/hipImportExternalMemory.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipImportExternalMemory_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  const auto storage = vkt.CreateMappedStorage<int>(1, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  auto desc = vkt.BuildMemoryDescriptor(storage.memory, sizeof(*storage.host_ptr));
  hipExternalMemory_t ext_memory;

// Disabled due to defect - EXSWHTEC-182
#if HT_NVIDIA
  SECTION("extMem_out == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalMemory(nullptr, &desc), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-183
#if HT_NVIDIA
  SECTION("memHandleDesc == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, nullptr), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-185
#if HT_NVIDIA
  SECTION("memHandleDesc.size == 0") {
    desc.size = 0;
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-186
#if HT_NVIDIA
  SECTION("Invalid memHandleDesc.flags") {
    desc.flags = 2;
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-184
#if HT_NVIDIA
  SECTION("Invalid memHandleDesc.type") {
    desc.type = static_cast<hipExternalMemoryHandleType>(-1);
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif

#ifdef _WIN32
  SECTION("memHandleDesc.handle == NULL") {
    desc.handle.win32.handle = NULL;
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif
}
