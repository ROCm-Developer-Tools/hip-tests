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
 * @addtogroup hipMemGetAllocationPropertiesFromHandle hipMemGetAllocationPropertiesFromHandle
 * @{
 * @ingroup VirtualTest
 * `hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
 * hipMemGenericAllocationHandle_t handle)` -
 * Retrieve the property structure of the given handle.
 */

/**
 * Test Description
 * ------------------------ 
 *  - Reads properties from allocated virtual memory and checks field values
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemGetAllocationPropertiesFromHandle.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemGetAllocationPropertiesFromHandle_Positive_Basic") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t size = 2 * 1024;
  VirtualMemoryGuard virtual_memory{size};

  hipMemAllocationProp property{};
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&property, virtual_memory.handle));

  REQUIRE(property.type == hipMemAllocationTypePinned);
  REQUIRE(property.location.id == 0);
  REQUIRE(property.location.type == hipMemLocationTypeDevice);
}

/**
 * Test Description
 * ------------------------ 
 *  - Passes invalid handle to API and expects that `hipErrorInvalidValue` is returned
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemGetAllocationPropertiesFromHandle.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemGetAllocationPropertiesFromHandle_Negative_Parameters") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t size = 2 * 1024;
  VirtualMemoryGuard virtual_memory{size};

  hipMemAllocationProp property{};
  hipMemGenericAllocationHandle_t handle{};
  HIP_CHECK_ERROR(hipMemGetAllocationPropertiesFromHandle(&property, handle), hipErrorInvalidValue);
}