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
 * @addtogroup hipUserObjectCreate hipUserObjectCreate
 * @{
 * @ingroup GraphTest
 * `hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
 * unsigned int initialRefcount, unsigned int flags)` -
 * Create an instance of userObject to manage lifetime of a resource.
 */

/* 1) Call hipUserObjectCreate once and release it by
      calling hipUserObjectRelease */
static void hipUserObjectCreate_Functional_1(void* object, void destroyObj(void*)) {
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRelease(hObject));
}

/**
 * Test Description
 * ------------------------
 *  - Creates user object from different types with ref count 1:
 *    -# When object is int
 *    -# When object is float
 *    -# When object is a class instance
 *    -# When structure is a structure instance
 *  - Checks that user object is not `nullptr`.
 *  - Releases user object with ref count 1.
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObjectCreate_Functional_1") {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyStructObj);
  }
}

/* 2) Call hipUserObjectCreate refCount as X and release it by
      calling hipUserObjectRelease with same refCount. */
static void hipUserObjectCreate_Functional_2(void* object, void destroyObj(void*)) {
  int refCount = 5;
  hipUserObject_t hObject;
  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyObj, refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRelease(hObject, refCount));
}

/**
 * Test Description
 * ------------------------
 *  - Creates user object from different types with ref count greater than 1:
 *    -# When object is int
 *    -# When object is float
 *    -# When object is a class instance
 *    -# When structure is a structure instance
 *  - Checks that user object is not `nullptr`.
 *  - Releases user object with ref count greater than 1.
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObjectCreate_Functional_2") {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyStructObj);
  }
}

/* 3) Call hipUserObjectCreate, retain it by calling hipUserObjectRetain
      and release it by calling hipUserObjectRelease twice. */
static void hipUserObjectCreate_Functional_3(void* object, void destroyObj(void*)) {
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
}

/**
 * Test Description
 * ------------------------
 *  - Creates user object from different types with ref count 1:
 *    -# When object is int
 *    -# When object is float
 *    -# When object is a class instance
 *    -# When structure is a structure instance
 *  - Checks that user object is not `nullptr`.
 *  - Retains user object once.
 *  - Releases user object twice with ref count 1.
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObjectCreate_Functional_3") {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyStructObj);
  }
}

/* 4) Call hipUserObjectCreate with refCount as X, retain it by calling
      hipUserObjectRetain with count as Y and release it by calling
      hipUserObjectRelease with count as X+Y. */
static void hipUserObjectCreate_Functional_4(void* object, void destroyObj(void*)) {
  int refCount = 5;
  int refCountRetain = 8;
  hipUserObject_t hObject;
  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyObj, refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject, refCountRetain));
  HIP_CHECK(hipUserObjectRelease(hObject, refCount + refCountRetain));
}

/**
 * Test Description
 * ------------------------
 *  - Creates user object from different types with ref count 5:
 *    -# When object is int
 *    -# When object is float
 *    -# When object is a class instance
 *    -# When structure is a structure instance
 *  - Checks that user object is not `nullptr`.
 *  - Retains user object with ref count 8.
 *  - Releases user object with ref count 13.
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObjectCreate_Functional_4") {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyStructObj);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid argument:
 *    -# When output pointer to the user object is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When object is `nullptr`
 *      - Expected output: return `hipSuccess`
 *    -# When destroy callback function handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When ref count is zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When ref count is INT_MAX
 *      - Expected output: return `hipSuccess`
 *    -# When flag is not valid
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObjectCreate_Negative") {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(
        hipUserObjectCreate(nullptr, object, destroyIntObj, 1, hipUserObjectNoDestructorSync),
        hipErrorInvalidValue);
  }
  SECTION("Pass object as nullptr") {
    HIP_CHECK(
        hipUserObjectCreate(&hObject, nullptr, destroyIntObj, 1, hipUserObjectNoDestructorSync));
  }
  SECTION("Pass Callback function as nullptr") {
    HIP_CHECK_ERROR(
        hipUserObjectCreate(&hObject, object, nullptr, 1, hipUserObjectNoDestructorSync),
        hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(
        hipUserObjectCreate(&hObject, object, destroyIntObj, 0, hipUserObjectNoDestructorSync),
        hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, INT_MAX,
                                  hipUserObjectNoDestructorSync));
  }
  SECTION("Pass flag other than hipUserObjectNoDestructorSync") {
    HIP_CHECK_ERROR(hipUserObjectCreate(&hObject, object, destroyIntObj, 1, hipUserObjectFlags(9)),
                    hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Creates new int object.
 *    - Expected output: return `hipSuccess`
 *  - Creates user object with 2 references.
 *    - Expected output: return `hipSuccess`
 *  - Releases more user objects (4) than created (2).
 *    - Expected output: return `hipSuccess`
 *  - Retains reference to a removed user object.
 *    - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/graph/hipUserObjectCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipUserObj_Negative_Test") {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;

  // Create a new hObject with 2 reference
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, 2, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  // Release more than created.
  HIP_CHECK(hipUserObjectRelease(hObject, 4));

  // Retain reference to a removed user object
  HIP_CHECK(hipUserObjectRetain(hObject, 1));
}
