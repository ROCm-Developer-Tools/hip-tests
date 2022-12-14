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

#include "virtual_memory_common.hh"

/**
 * @addtogroup hipMemGetAccess hipMemGetAccess
 * @{
 * @ingroup VirtualTest
 * `hipMemGetAccess(unsigned long long* flags, 
 * const hipMemLocation* location, void* ptr)` -
 * Get the access flags set for the given location and ptr.
 */

/**
 * Test Description
 * ------------------------ 
 *  - Creates new virual memory allocation and checks access flags.
 * Test source
 * ------------------------ 
 *    - unit/virtualMemory/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------ 
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemGetAccess_Positive_Basic") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t size = 2 * 1024;
  unsigned long long flags{0};
  VirtualMemoryGuard virtual_memory{size};

  hipMemLocation location{};
  location.id = 0;
  location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipMemGetAccess(&flags, &location, virtual_memory.virtual_memory_ptr));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
}

/**
 * Test Description
 * ------------------------ 
 *  - Passes invalid parameters and checks behaviour:
 *    -# When device is invalid, ordinal is out of range
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When virtual memory pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memory location pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemGetAccess_Negative_Parameters") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t size = 2 * 1024;
  unsigned long long flags{0};
  VirtualMemoryGuard virtual_memory{size};

  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;

  SECTION("invalid device") {
    location.id = HipTest::getDeviceCount();
    HIP_CHECK_ERROR(hipMemGetAccess(&flags, &location, virtual_memory.virtual_memory_ptr), hipErrorInvalidDevice);
  }

  SECTION("invalid virtual memory pointer") {
    location.id = 0;
    HIP_CHECK_ERROR(hipMemGetAccess(&flags, &location, nullptr), hipErrorInvalidValue);
  }

  SECTION("invalid memory location pointer") {
    HIP_CHECK_ERROR(hipMemGetAccess(&flags, nullptr, virtual_memory.virtual_memory_ptr), hipErrorInvalidValue);
  }
}

/**
 * End doxygen group hipMemGetAccess.
 * @}
 */

/**
 * @addtogroup hipMemSetAccess hipMemSetAccess
 * @{
 * @ingroup VirtualTest
 * `hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc,
 * size_t count)` -
 * Set the access flags for each location specified in desc for the
 * given virtual address range.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemVmm_Positive_OneToOne_Mapping
 *  - @ref Unit_hipMemVmm_Positive_OneToN_Mapping
 */

/**
 * Test Description
 * ------------------------ 
 *  - Passes invalid parameters and checks behaviour:
 *    -# Sets access for the larger virtual memory chunk than allocated
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Wrong access description array size
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Device ID is out of range of available IDs
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Pointer to the access array is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Pointer to the virtual memory is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemSetAccess_Negative_Parameters") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  size_t allocation_size = calculate_allocation_size(2 * 1024);
  hipMemGenericAllocationHandle_t handle{};

  hipMemAllocationProp properties{};
  properties.type = hipMemAllocationTypePinned;
  properties.location.id = 0;
  properties.location.type = hipMemLocationTypeDevice;

  hipMemAccessDesc access{};
  access.flags = hipMemAccessFlagsProtReadWrite;
  access.location = properties.location;

  HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
  HIP_CHECK(hipMemCreate(&handle, allocation_size, &properties, 0));
  HIP_CHECK(hipMemMap(virtual_memory_ptr, allocation_size, 0, handle, 0));

  SECTION("set access for larger virtual memory than allocated") {
    HIP_CHECK_ERROR(hipMemSetAccess(virtual_memory_ptr, 2 * allocation_size, &access, 1), hipErrorInvalidValue);
  }

  SECTION("wrong access description array size") {
    HIP_CHECK_ERROR(hipMemSetAccess(virtual_memory_ptr, allocation_size, &access, 0), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemSetAccess(virtual_memory_ptr, allocation_size, &access, 2), hipErrorInvalidValue);
  }

  SECTION("wrong device ID") {
    access.location.id = HipTest::getDeviceCount();
    HIP_CHECK_ERROR(hipMemSetAccess(virtual_memory_ptr, allocation_size, &access, 1), hipErrorInvalidValue);
  }

  SECTION("access is nullptr") {
    HIP_CHECK_ERROR(hipMemSetAccess(virtual_memory_ptr, allocation_size, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("virtual memory pointer is nullptr") {
    HIP_CHECK_ERROR(hipMemSetAccess(nullptr, allocation_size, &access, 1), hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemUnmap(virtual_memory_ptr, allocation_size));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
}

/**
 * Test Description
 * ------------------------ 
 *  - Sets access to the virtual memory before it is mapped to the physical memory
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemSetAccess_Negative_SetBeforeMapped") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  size_t allocation_size = calculate_allocation_size(2 * 1024);
  hipMemGenericAllocationHandle_t handle{};

  hipMemAllocationProp properties{};
  properties.type = hipMemAllocationTypePinned;
  properties.location.id = 0;
  properties.location.type = hipMemLocationTypeDevice;

  hipMemAccessDesc access{};
  access.flags = hipMemAccessFlagsProtReadWrite;
  access.location = properties.location;

  HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
  HIP_CHECK(hipMemCreate(&handle, allocation_size, &properties, 0));
  HIP_CHECK_ERROR(hipMemSetAccess(virtual_memory_ptr, allocation_size, &access, 1), hipErrorInvalidValue);
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
}