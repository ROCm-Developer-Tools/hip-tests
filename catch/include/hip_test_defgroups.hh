/*
Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

// Test groups are named based on the group names from hip_api_runtime.h, with adding "Test" suffix

/**
 * @defgroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 * @}
 */

/**
 * @defgroup GraphTest Graph Management
 * @{
 * This section describes the graph management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup MemoryTest Memory Management
 * @{
 * This section describes tests for the memory management functions of HIP runtime API.
 */

/**
 * @addtogroup hipMemset hipMemset
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemset") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemset") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemset") {}
/**
 * End doxygen group hipMemset.
 * @}
 */

/**
 * @addtogroup hipMemsetD32 hipMemsetD32
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD32") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD32") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") {}
/**
 * End doxygen group hipMemsetD32.
 * @}
 */

/**
 * @addtogroup hipMemsetD16 hipMemsetD16
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD16") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD16") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") {}
/**
 * End doxygen group hipMemsetD16.
 * @}
 */

/**
 * @addtogroup hipMemsetD8 hipMemsetD8
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD8") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD8") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") {}
/**
 * End doxygen group hipMemsetD8.
 * @}
 */

/**
 * End doxygen group MemoryTest.
 * @}
 */

/**
 * @defgroup MemoryMTest Managed Memory
 * @{
 * This section describes tests for the managed memory management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup PeerToPeerTest PeerToPeer Device Memory Access
 * @{
 *  @warning PeerToPeer support is experimental.
 *  This section describes tests for the PeerToPeer device memory access functions of HIP runtime API.
 * @}
 */
