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

#include <limits>
#include "DriverContext.hh"
#include "hipArrayCommon.hh"
#include "hip_array_common.hh"
#include "hip_test_common.hh"

/**
 * @addtogroup hipArray3DCreate hipArray3DCreate
 * @{
 * @ingroup MemoryTest
 * `hipArray3DCreate(hipArray** array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray)` -
 * Create a 3D array.
 */

namespace {
void checkArrayIsExpected(const hiparray array, const HIP_ARRAY3D_DESCRIPTOR& expected_desc) {
// hipArray3DGetDescriptor doesn't currently exist (EXSWCPHIPT-87)
#if HT_AMD
  std::ignore = array;
  std::ignore = expected_desc;
#else
  CUDA_ARRAY3D_DESCRIPTOR queried_desc;
  cuArray3DGetDescriptor(&queried_desc, array);

  REQUIRE(queried_desc.Width == expected_desc.Width);
  REQUIRE(queried_desc.Height == expected_desc.Height);
  REQUIRE(queried_desc.Depth == expected_desc.Depth);
  REQUIRE(queried_desc.Format == expected_desc.Format);
  REQUIRE(queried_desc.NumChannels == expected_desc.NumChannels);
  REQUIRE(queried_desc.Flags == expected_desc.Flags);
#endif
}

void testInvalidDescription(HIP_ARRAY3D_DESCRIPTOR desc) {
  hiparray array;
  HIP_CHECK_ERROR(hipArray3DCreate(&array, &desc), hipErrorInvalidValue);
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Validates handling that 3D array is created successfully for
 *    different types of data and supported flags.
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipArray3DCreate_happy", "", char, uchar2, uint2, int4, short4, float,
                   float2, float4) {
  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
#if HT_AMD
  desc.Flags = 0;
#else
  desc.Flags = GENERATE(0, hipArraySurfaceLoadStore, hipArrayTextureGather);
#endif

  constexpr size_t size = 64;

  std::vector<hipExtent> extents;
  extents.reserve(3);
  extents.push_back({size, size, 0});  // 2D array
  if (desc.Flags != hipArrayTextureGather) {
    extents.push_back({size, 0, 0});        // 1D array
    extents.push_back({size, size, size});  // 3D array
  };

  for (auto& extent : extents) {
    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    hiparray array;
    HIP_CHECK(hipArray3DCreate(&array, &desc));
    checkArrayIsExpected(array, desc);
    HIP_CHECK(hipArrayDestroy(array));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validate that array 3D is created successfully for different
 *    types of data when its width/height/depth are set to maximal size.
 *  - Maximal size corresponds to maximal texture width/height/width.
 *  - Test can fail with `hipErrorOutMemory`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipArray3DCreate_MaxTexture", "", int, uint4, short, ushort2,
                   unsigned char, float, float4) {
  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  hiparray array;
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
#if HT_AMD
  desc.Flags = 0;
#else
  desc.Flags = GENERATE(0, hipArraySurfaceLoadStore);
  if (desc.Flags == hipArraySurfaceLoadStore) {
    HipTest::HIP_SKIP_TEST("EXSWCPHIPT-58");
    return;
  }
#endif
  CAPTURE(desc.Flags);

  const Sizes sizes(desc.Flags);
  CAPTURE(sizes.max1D, sizes.max2D, sizes.max3D);

  const size_t s = 64;
  SECTION("Happy") {
    // stored in a vector so some values can be ifdef'd out
    std::vector<hipExtent> extentsToTest{
        make_hipExtent(sizes.max1D, 0, 0),                              // 1D max
        make_hipExtent(sizes.max2D[0], s, 0),                           // 2D max width
        make_hipExtent(s, sizes.max2D[1], 0),                           // 2D max height
        make_hipExtent(sizes.max2D[0], sizes.max2D[1], 0),              // 2D max
        make_hipExtent(sizes.max3D[0], s, s),                           // 3D max width
        make_hipExtent(s, sizes.max3D[1], s),                           // 3D max height
        make_hipExtent(s, s, sizes.max3D[2]),                           // 3D max depth
        make_hipExtent(s, sizes.max3D[1], sizes.max3D[2]),              // 3D max height and depth
        make_hipExtent(sizes.max3D[0], s, sizes.max3D[2]),              // 3D max width and depth
        make_hipExtent(sizes.max3D[0], sizes.max3D[1], s),              // 3D max width and height
        make_hipExtent(sizes.max3D[0], sizes.max3D[1], sizes.max3D[2])  // 3D max
    };
    const auto extent =
        GENERATE_COPY(from_range(std::begin(extentsToTest), std::end(extentsToTest)));

    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    auto maxArrayCreateError = hipArray3DCreate(&array, &desc);
    // this can try to alloc many GB of memory, so out of memory is acceptable
    if (maxArrayCreateError == hipErrorOutOfMemory) return;
    HIP_CHECK(maxArrayCreateError);
    checkArrayIsExpected(array, desc);
    HIP_CHECK(hipArrayDestroy(array));
  }
  SECTION("Negative") {
    std::vector<hipExtent> extentsToTest {
      make_hipExtent(sizes.max1D + 1, 0, 0),                          // 1D max
          make_hipExtent(sizes.max2D[0] + 1, s, 0),                   // 2D max width
          make_hipExtent(s, sizes.max2D[1] + 1, 0),                   // 2D max height
          make_hipExtent(sizes.max2D[0] + 1, sizes.max2D[1] + 1, 0),  // 2D max
          make_hipExtent(sizes.max3D[0] + 1, s, s),                   // 3D max width
          make_hipExtent(s, sizes.max3D[1] + 1, s),                   // 3D max height
#if !HT_NVIDIA                                       // leads to hipSuccess on NVIDIA
          make_hipExtent(s, s, sizes.max3D[2] + 1),  // 3D max depth
#endif
          make_hipExtent(s, sizes.max3D[1] + 1, sizes.max3D[2] + 1),  // 3D max height and depth
          make_hipExtent(sizes.max3D[0] + 1, s, sizes.max3D[2] + 1),  // 3D max width and depth
          make_hipExtent(sizes.max3D[0] + 1, sizes.max3D[1] + 1, s),  // 3D max width and height
          make_hipExtent(sizes.max3D[0] + 1, sizes.max3D[1] + 1, sizes.max3D[2] + 1)  // 3D max
    };
    const auto extent =
        GENERATE_COPY(from_range(std::begin(extentsToTest), std::end(extentsToTest)));

    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    HIP_CHECK_ERROR(hipArray3DCreate(&array, &desc), hipErrorInvalidValue);
  }
}

#if HT_NVIDIA
constexpr std::array<unsigned int, 9> validFlags{
    0,
    hipArraySurfaceLoadStore,
    hipArrayLayered,
    hipArrayLayered | hipArraySurfaceLoadStore,
    hipArrayCubemap,
    hipArrayCubemap | hipArrayLayered,
    hipArrayCubemap | hipArraySurfaceLoadStore,
    hipArrayCubemap | hipArrayLayered | hipArraySurfaceLoadStore,
    hipArrayTextureGather};
#else
constexpr std::array<unsigned int, 5> validFlags{
    0, hipArrayCubemap, hipArrayCubemap | hipArrayLayered,
    hipArrayCubemap | hipArraySurfaceLoadStore,
    hipArrayCubemap | hipArrayLayered | hipArraySurfaceLoadStore};
#endif

constexpr HIP_ARRAY3D_DESCRIPTOR defaultDescriptor(unsigned int flags, size_t size) {
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Format = HIP_AD_FORMAT_FLOAT;
  desc.NumChannels = 4;
  desc.Flags = flags;
  desc.Width = size;
  desc.Height = size;
  desc.Depth = size;

#if HT_NVIDIA
  if (flags == hipArrayTextureGather) {
    desc.Depth = 0;
  }
#endif
  return desc;
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when the array is `nullptr`
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_NullArrayPtr") {
  auto desc = defaultDescriptor(0, 64);

  DriverContext ctx;
  HIP_CHECK_ERROR(hipArray3DCreate(nullptr, &desc), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when array descriptor is `nullptr`
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_NullDescPtr") {
  DriverContext ctx;
  hiparray array;
  HIP_CHECK_ERROR(hipArray3DCreate(&array, nullptr), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when width is zero
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_ZeroWidth") {
  DriverContext ctx;

  unsigned int flags = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  auto desc = defaultDescriptor(flags, 6);
  desc.Width = 0;
  CAPTURE(desc.Flags);

  testInvalidDescription(desc);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when height is zero
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_ZeroHeight") {
  DriverContext ctx;

  unsigned int flags = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  auto desc = defaultDescriptor(flags, 6);
#if HT_NVIDIA
  std::array<unsigned int, 2> exceptions{hipArrayLayered,
                                         hipArrayLayered | hipArraySurfaceLoadStore};
#else
  std::array<unsigned int, 0> exceptions{};
#endif
  desc.Height = 0;

  if (std::find(std::begin(exceptions), std::end(exceptions), desc.Flags) == std::end(exceptions)) {
    // flag is not in list of exceptions
    testInvalidDescription(desc);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when channel descriptor format is not valid
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_InvalidFormat") {
  DriverContext ctx;

  unsigned int flags = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  auto desc = defaultDescriptor(flags, 6);

  desc.Format = static_cast<hipArray_Format>(0xDEADBEEF);
  REQUIRE(std::find(std::begin(driverFormats), std::end(driverFormats), desc.Format) ==
          std::end(driverFormats));

  testInvalidDescription(desc);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when channel number in descriptor is not valid
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_NumChannels") {
  DriverContext ctx;
  unsigned int flags = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  auto desc = defaultDescriptor(flags, 6);
  desc.NumChannels = GENERATE(0, 3, 5);

  testInvalidDescription(desc);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when flag values in descriptor are not valid:
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_InvalidFlags") {
  DriverContext ctx;

  // FIXME: use the same flags for both tests when the values exist for hip
#if HT_NVIDIA
  unsigned int flags = GENERATE(0xDEADBEEF, hipArrayTextureGather | hipArraySurfaceLoadStore,
                                hipArrayTextureGather | hipArrayCubemap,
                                hipArrayTextureGather | hipArraySurfaceLoadStore | hipArrayCubemap);
#else
  unsigned int flags = 0xDEADBEEF;
#endif

  CAPTURE(flags);

  auto desc = defaultDescriptor(flags, 6);


  REQUIRE(std::find(std::begin(validFlags), std::end(validFlags), desc.Flags) ==
          std::end(validFlags));

  testInvalidDescription(desc);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when height/width/depth values are set to maximal
 *    numerical values
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipArray3DCreate_Negative_NumericLimit") {
  DriverContext ctx;

  unsigned int flags = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  auto desc = defaultDescriptor(flags, std::numeric_limits<size_t>::max());

  testInvalidDescription(desc);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling when descriptor for array with flag set to texture gather
 *    is set to represent 1D or 3D array for various types, which is not valid.
 *    - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DCreate.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipArray3DCreate_Negative_Non2DTextureGather", "", char, uint2, int4,
                   float2, float4) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("Texture Gather arrays not supported using AMD backend");
  return;
#endif
  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Flags = hipArrayTextureGather;

  constexpr size_t size = 64;

  std::array<hipExtent, 2> extents{
      make_hipExtent(size, 0, 0),        // 1D array
      make_hipExtent(size, size, size),  // 3D array
  };

  for (auto& extent : extents) {
    desc.Width = extent.width;
    desc.Height = extent.height;
    desc.Depth = extent.depth;

    CAPTURE(desc.Width, desc.Height, desc.Depth);

    testInvalidDescription(desc);
  }
}
