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

#include "user_object_common.hh"

/**
 * @addtogroup hipUserObjectRetain hipUserObjectRetain
 * @{
 * @ingroup GraphTest
 * `hipUserObjectRetain(hipUserObject_t object, unsigned int count __dparm(1))` -
 * Retain number of references to resource.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipUserObjectCreate_Functional_3
 *  - @ref Unit_hipUserObjectCreate_Functional_4
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When user object handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When ref count is zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When ref count is INT_MAX
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectRetain.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObjectRetain_Negative") {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(hipUserObjectRetain(nullptr, 1), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(hipUserObjectRetain(hObject, 0), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as INT_MAX") { HIP_CHECK(hipUserObjectRetain(hObject, INT_MAX)); }
}