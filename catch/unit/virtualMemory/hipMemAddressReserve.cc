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
 * @addtogroup hipMemAddressReserve hipMemAddressReserve
 * @{
 * @ingroup VirtualTest
 * `hipMemAddressReserve(void** ptr, size_t size, size_t alignment,
 * void* addr, unsigned long long flags)` -
 * Reserves an address range
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemVmm_Positive_OneToOne_Mapping
 *  - @ref Unit_hipMemVmm_Positive_OneToN_Mapping
 */

/**
 * Test Description
 * ------------------------ 
 *  - Tries to reserve upper limit of type `size_t` bytes of virtual memory
 * Test source
 * ------------------------ 
 *  - unit/virtualMemory/hipMemAddressReserve.cc
 * Test requirements
 * ------------------------ 
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemAddressReserve_Negative_Basic") {
  if (!is_virtual_memory_management_supported()) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
	void* virtual_memory_ptr{nullptr};
	size_t allocation_size = calculate_allocation_size(std::numeric_limits<size_t>::max());
	HIP_CHECK_ERROR(hipMemAddressReserve(&virtual_memory_ptr, allocation_size, 0, nullptr, 0), hipErrorInvalidValue);
}