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
 *  - Maps physical address to the reserved virutal memory range
 *    -# Writes data to the virtual memory range
 *    -# Checks that the data can be acquired and is valid
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemVmm.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemVmm_Positive_OneToOne_Mapping") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t size = 4 * 1024;
  VirtualMemoryGuard virtual_memory{size};

  hipDeviceptr_t device_memory_ptr = reinterpret_cast<hipDeviceptr_t>(virtual_memory.virtual_memory_ptr);
  HIP_CHECK(hipMemsetD32(device_memory_ptr, 0xDEADBEAF, size/4));
  std::vector<unsigned int> values(size/4);
  HIP_CHECK(hipMemcpy(&values[0], virtual_memory.virtual_memory_ptr, size, hipMemcpyDeviceToHost));

  for (const auto& value: values) {
    REQUIRE(value == 0xDEADBEAF);
  }
}

/**
 * Test Description
 * ------------------------ 
 *  - Maps one physical address to two different reserved virutal memory ranges
 *    -# Writes data to the first virtual memory range
 *    -# Checks that the data can be acquired from the second virtual memory range
 *    -# Expects that the first virtual memory range and the
 *       second virtual memory range contain the same data
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemVmm.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemVmm_Positive_OneToN_Mapping") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t size = sizeof(unsigned int) * 1024;
  VirtualMemoryGuard virtual_memory_A{size};
  VirtualMemoryGuard virtual_memory_B{size, 0, &virtual_memory_A.handle};

  hipDeviceptr_t device_memory_ptr = reinterpret_cast<hipDeviceptr_t>(virtual_memory_A.virtual_memory_ptr);
  HIP_CHECK(hipMemsetD32(device_memory_ptr, 0xDEADBEAF, size/sizeof(unsigned int)));
  std::vector<unsigned int> values(size/sizeof(unsigned int));
  HIP_CHECK(hipMemcpy(&values[0], virtual_memory_B.virtual_memory_ptr, size, hipMemcpyDeviceToHost));

  for (const auto& value: values) {
    REQUIRE(value == 0xDEADBEAF);
  }
}