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
 * @addtogroup hipMemUnmap hipMemUnmap
 * @{
 * @ingroup VirtualTest
 * `hipMemUnmap(void* ptr, size_t size)` -
 * Unmap memory allocation of a given address range.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemVmm_Positive_OneToOne_Mapping
 *  - @ref Unit_hipMemVmm_Positive_OneToN_Mapping
 */

/**
 * Test Description
 * ------------------------ 
 *  - Passes invalid parameters and checks behaviour:
 *    -# The size of unmapped virtual memory is smaller than allocated
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Virtual memory pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemUnmap.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemUnmap_Negative_Parameters") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  void* virtual_memory_ptr{nullptr};
  size_t allocation_size = 2 * calculate_allocation_size(2 * 1024);
  hipMemGenericAllocationHandle_t handle{};

  hipMemAllocationProp properties{};
  properties.type = hipMemAllocationTypePinned;
  properties.location.id = 0;
  properties.location.type = hipMemLocationTypeDevice;

  hipMemAccessDesc access{};
  access.flags = hipMemAccessFlagsProtReadWrite;
  access.location = properties.location;

  SECTION("unmap size is smaller than reserved") {
    HIP_CHECK(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0));
    HIP_CHECK(hipMemCreate(&handle, allocation_size, &properties, 0));
    HIP_CHECK(hipMemMap(virtual_memory_ptr, allocation_size, 0, handle, 0));

    HIP_CHECK_ERROR(hipMemUnmap(virtual_memory_ptr, allocation_size/2), hipErrorInvalidValue);

    HIP_CHECK(hipMemUnmap(virtual_memory_ptr, allocation_size));
    HIP_CHECK(hipMemRelease(handle));
    HIP_CHECK(hipMemAddressFree(virtual_memory_ptr, allocation_size));
  }

  SECTION("virtual memory pointer is nullptr") {
    HIP_CHECK_ERROR(hipMemUnmap(nullptr, allocation_size), hipErrorInvalidValue);
  }  
}