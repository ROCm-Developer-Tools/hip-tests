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
 * @addtogroup hipMemMap hipMemMap
 * @{
 * @ingroup VirtualTest
 * `hipError_t hipMemMap(void* ptr, size_t size, size_t offset,
 * hipMemGenericAllocationHandle_t handle, unsigned long long flags)` -
 * Maps an allocation handle to a reserved virtual address range.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemVmm_Positive_OneToOne_Mapping
 *  - @ref Unit_hipMemVmm_Positive_OneToN_Mapping
 */

/**
 * Test Description
 * ------------------------ 
 *  - Call the API in different negative scenarios regarding parameters:
 *    -# When larger virtual memory than allocated is mapped
 *      - Expected output: return hipErrorNotSupported
 *    -# When virtual memory pointer is freed
 *      - Expected output: return hipErrorInvalidValue
 *    -# When physical handle is released
 *      - Expected output: return hipErrorInvalidValue
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemMap.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemMap_Negative_Basic") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t allocation_size = calculate_allocation_size(2 * 1024);
  void* virtual_memory_ptr{};
  hipMemGenericAllocationHandle_t handle{};
  hipMemAllocationProp properties{};
  properties.type = hipMemAllocationTypePinned;
  properties.location.id = 0;
  properties.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
  HIP_CHECK(hipMemCreate(&handle, allocation_size, &properties, 0));

  SECTION("map larger virtual memory than allocated") {
    HIP_CHECK_ERROR(hipMemMap(virtual_memory_ptr, 2 * allocation_size, 0, handle, 0), hipErrorNotSupported);
    HIP_CHECK(hipMemRelease(handle));
    HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
  }

  SECTION("virtual memory pointer is freed") {
    HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
    HIP_CHECK_ERROR(hipMemMap(virtual_memory_ptr, allocation_size, 0, handle, 0), hipErrorInvalidValue);
    HIP_CHECK(hipMemRelease(handle));
  }

  SECTION("handle is released") {
    HIP_CHECK(hipMemRelease(handle));
    HIP_CHECK_ERROR(hipMemMap(virtual_memory_ptr, allocation_size, 0, handle, 0), hipErrorInvalidValue);
    HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
  }
}