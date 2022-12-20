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

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipModuleGetFunction hipModuleGetFunction
 * @{
 * @ingroup ModuleTest
 * `hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname)` -
 * Function with kname will be extracted if present in module.
 */

static hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_function_module.code");
  return mg.module();
}

/**
 * Test Description
 * ------------------------
 *  - Get function from the loaded module.
 *  - Checks that the function is not `nullptr`.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleGetFunction.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleGetFunction_Positive_Basic") {
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, GetModule(), "GlobalKernel"));
  REQUIRE(kernel != nullptr);
}

/**
 * Test Description
 * ------------------------
 *  - Verifies handling of invalid arguments:
 *    -# When output pointer to the function is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When module pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the function name is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the function name is an empty string
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the function name is not existing function
 *      - Expected output: return `hipErrorNotFound`
 *    -# When the function name is __device__ function
 *      - Expected output: return `hipErrorNotFound`
 * Test source
 * ------------------------
 *  - unit/module/hipModuleGetFunction.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleGetFunction_Negative_Parameters") {
  hipFunction_t kernel = nullptr;

  SECTION("function == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunction(nullptr, GetModule(), "GlobalKernel"),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-154
#if HT_NVIDIA
  SECTION("module == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, nullptr, "GlobalKernel"),
                    hipErrorInvalidResourceHandle);
  }
#endif

  SECTION("kname == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, GetModule(), nullptr), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-155
#if HT_NVIDIA
  SECTION("kname == empty string") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, GetModule(), ""), hipErrorInvalidValue);
  }
#endif

  SECTION("kname == non existent kernel") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, GetModule(), "NonExistentKernel"),
                    hipErrorNotFound);
  }

  SECTION("kname == __device__ kernel") {
    HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, GetModule(), "DeviceKernel"), hipErrorNotFound);
  }
}
