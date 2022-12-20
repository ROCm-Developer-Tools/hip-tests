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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipModuleUnload hipModuleUnload
 * @{
 * @ingroup ModuleTest
 * `hipModuleUnload(hipModule_t module)` -
 * Frees the module.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipModuleLoad_Positive_Basic
 *  - @ref Unit_hipModuleLoadData_Positive_Basic
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When module is `nullptr`
 *      - Expected output: return `hipErrorInvalidResourceHandle`
 *    -# When module is already unloaded
 *      - Expected output: return `hipErrorNotFound`
 * Test source
 * ------------------------
 *  - unit/module/hipModuleUnload.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleUnload_Negative_Parameters") {
  HIP_CHECK(hipFree(nullptr));

// Disabled for AMD due to defect - EXSWHTEC-152
#if HT_NVIDIA
  SECTION("module == nullptr") {
    HIP_CHECK_ERROR(hipModuleUnload(nullptr), hipErrorInvalidResourceHandle);
  }
#endif

// Causes CUDA to segfault
#if HT_AMD
  SECTION("Double unload") {
    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK_ERROR(hipModuleUnload(module), hipErrorNotFound);
  }
#endif
}
