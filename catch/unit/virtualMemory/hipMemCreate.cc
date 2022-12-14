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
 * @addtogroup hipMemCreate hipMemCreate
 * @{
 * @ingroup VirtualTest
 * `hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
 * const hipMemAllocationProp* prop, unsigned long long flags)` -
 * Creates a memory allocation described by the properties and size
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemVmm_Positive_OneToOne_Mapping
 *  - @ref Unit_hipMemVmm_Positive_OneToN_Mapping
 */

/**
 * Test Description
 * ------------------------ 
 *  - Call the API in different negative scenarios regarding parameters:
 *    -# When allocation size is not the multiple of granularity
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ordinal within properties is out of available range
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When properties are `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemCreate.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemCreate_Negative_Parameters") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t allocation_size = calculate_allocation_size(2 * 1024);
  hipMemGenericAllocationHandle_t handle{};

  hipMemAllocationProp properties{};
  properties.type = hipMemAllocationTypePinned;
  properties.location.id = 0;
  properties.location.type = hipMemLocationTypeDevice;

  SECTION("invalid allocation size") {
    HIP_CHECK_ERROR(hipMemCreate(&handle, 2 * 1024, &properties, 0), hipErrorInvalidValue);
  }

  SECTION("device ordinal number out of range") {
    properties.location.id = HipTest::getDeviceCount();
    HIP_CHECK_ERROR(hipMemCreate(&handle, allocation_size, &properties, 0), hipErrorInvalidDevice);
  }

  SECTION("handle is nullptr") {
    HIP_CHECK_ERROR(hipMemCreate(nullptr, allocation_size, &properties, 0), hipErrorInvalidValue);
  }

  SECTION("allocation properties are nullptr") {
    HIP_CHECK_ERROR(hipMemCreate(&handle, allocation_size, nullptr, 0), hipErrorInvalidValue);
  }
}