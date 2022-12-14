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
 * @addtogroup hipMemAddressFree hipMemAddressFree
 * @{
 * @ingroup VirtualTest
 * `hipMemAddressFree(void* devPtr, size_t size)` -
 * Frees an address range reservation made via hipMemAddressReserve
 */

/**
 * Test Description
 * ------------------------ 
 *  - Allocates and frees regular chunk of virtual memory
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemAddressFree.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemAddressFree_Positive_Basic") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  int size_mult = GENERATE(8, 32, 128);
  size_t allocation_size = calculate_allocation_size(size_mult * 1024);

  HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
  HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
}

/**
 * Test Description
 * ------------------------ 
 *  - Checks different invalid scenarios with freeing virtual memory:
 *    -# Tries to free more virtual memory than allocated
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Tries to free virtual memory with `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemAddressFree.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemAddressFree_Negative_Parameters") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  size_t allocation_size = calculate_allocation_size(2 * 1024);

  SECTION("free more memory than reserved") {
    HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
    HIP_CHECK_ERROR(hipMemAddressFree(virtual_memory_ptr, 2 * allocation_size), hipErrorInvalidValue);
    HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
  }

  SECTION("invalid virtual memory pointer") {
    void* virutal_memory_ptr{nullptr};
    HIP_CHECK_ERROR(hipMemAddressFree(virtual_memory_ptr, allocation_size), hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------ 
 *  - Frees virtual memory two times in a row
 *    -# Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemAddressFree.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemAddressFree_Negative_FreeMemoryTwoTimes") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  size_t allocation_size = calculate_allocation_size(2 * 1024);

  HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
  HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
  HIP_CHECK_ERROR(hipMemAddressFree(virtual_memory_ptr, allocation_size), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------ 
 *  - Frees the virtual memory before it is unmapped from physical memory
 *    -# Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemAddressFree.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemAddressFree_Negative_FreeBeforeUnmap") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  hipMemGenericAllocationHandle_t handle{};
  hipMemAllocationProp properties{};
  properties.type = hipMemAllocationTypePinned;
  properties.location.id = 0;
  properties.location.type = hipMemLocationTypeDevice;
  size_t allocation_size = calculate_allocation_size(2 * 1024);

  HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
  HIP_CHECK(hipMemCreate(&handle, allocation_size, &properties, 0));
  HIP_CHECK(hipMemMap(virtual_memory_ptr, allocation_size, 0, handle, 0));

  HIP_CHECK_ERROR(hipMemAddressFree(virtual_memory_ptr, allocation_size), hipErrorInvalidValue);

  HIP_CHECK(hipMemUnmap(virtual_memory_ptr, allocation_size));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
}