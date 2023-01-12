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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>


class TexObjectTestWrapper {
 private:
  float* host_data_;
  bool ommit_destroy_;

 public:
  hipTextureObject_t texture_object = 0;
  HIP_RESOURCE_DESC res_desc;
  HIP_TEXTURE_DESC tex_desc;
  HIP_RESOURCE_VIEW_DESC res_view_desc;
  HIP_ARRAY_DESCRIPTOR array_desc;
  hiparray array_member;
  size_t size; /* size in bytes*/
  int width;   /* width in elements */

  TexObjectTestWrapper(bool useResourceViewDescriptor, bool ommitDestroy = false)
      : ommit_destroy_(ommitDestroy), width(128) {
    int i;
    size = width * sizeof(float);

    host_data_ = (float*)malloc(size);
    memset(host_data_, 0, size);

    for (i = 0; i < width; i++) {
      host_data_[i] = i;
    }

    memset(&array_desc, 0, sizeof(array_desc));
    array_desc.Format = HIP_AD_FORMAT_FLOAT;
    array_desc.NumChannels = 1;
    array_desc.Width = width;
    array_desc.Height = 0;

    HIP_CHECK(hipArrayCreate(&array_member, &array_desc));
    HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray*>(array_member), 0, host_data_, size));

    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = array_member;
    res_desc.flags = 0;

    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.filterMode = HIP_TR_FILTER_MODE_POINT;
    tex_desc.flags = 0;

    memset(&res_view_desc, 0, sizeof(res_view_desc));


    if (useResourceViewDescriptor) {
#if HT_AMD
      res_view_desc.format = HIP_RES_VIEW_FORMAT_FLOAT_1X32;
      res_view_desc.width = size;
#else
      /* Resource View Descriptors are not supported on NVIDIA currently */
      useResourceViewDescriptor = false;
#endif
    }


    HIP_CHECK(hipTexObjectCreate(&texture_object, &res_desc, &tex_desc,
                                 useResourceViewDescriptor ? &res_view_desc : nullptr));
  }

  ~TexObjectTestWrapper() {
    if (!ommit_destroy_) {
      HIP_CHECK(hipTexObjectDestroy(texture_object));
    }
    HIP_CHECK(hipArrayDestroy(array_member));
    free(host_data_);
  }
};

/* hipTexObjectGetResourceDesc tests */

TEST_CASE("Unit_hipGetTexObjectResourceDesc_positive") {
  CHECK_IMAGE_SUPPORT;

  TexObjectTestWrapper tex_obj_wrapper(false);

  HIP_RESOURCE_DESC check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  HIP_CHECK(hipTexObjectGetResourceDesc(&check_desc, tex_obj_wrapper.texture_object));

  REQUIRE(check_desc.resType == tex_obj_wrapper.res_desc.resType);
  REQUIRE(check_desc.res.array.hArray == tex_obj_wrapper.res_desc.res.array.hArray);
}


TEST_CASE("Unit_hipGetTexObjectResourceDesc_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

  TexObjectTestWrapper tex_obj_wrapper(false);

  HIP_RESOURCE_DESC check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceDesc(nullptr, tex_obj_wrapper.texture_object),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceDesc(&check_desc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }
}


/* hipTexObjectGetResourceViewDesc tests */
#if HT_AMD
TEST_CASE("Unit_hipGetTexObjectResourceViewDesc_positive") {
  CHECK_IMAGE_SUPPORT;

  TexObjectTestWrapper tex_obj_wrapper(true);

  HIP_RESOURCE_VIEW_DESC check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  HIP_CHECK(hipTexObjectGetResourceViewDesc(&check_desc, tex_obj_wrapper.texture_object));

  REQUIRE(check_desc.format == tex_obj_wrapper.res_view_desc.format);
  REQUIRE(check_desc.width == tex_obj_wrapper.res_view_desc.width);
}
#endif

#if HT_AMD
TEST_CASE("Unit_hipGetTexObjectResourceViewDesc_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;
  TexObjectTestWrapper tex_obj_wrapper(true);

  HIP_RESOURCE_VIEW_DESC check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceViewDesc(nullptr, tex_obj_wrapper.texture_object),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(
        hipTexObjectGetResourceViewDesc(&check_desc, static_cast<hipTextureObject_t>(0)),
        hipErrorInvalidValue);
  }
}
#endif

/* hipTexObjectGetTextureDesc tests */


TEST_CASE("Unit_hipGetTexObjectTextureDesc_positive") {
  CHECK_IMAGE_SUPPORT;

  TexObjectTestWrapper tex_obj_wrapper(false);

  HIP_TEXTURE_DESC check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  HIP_CHECK(hipTexObjectGetTextureDesc(&check_desc, tex_obj_wrapper.texture_object));

  REQUIRE(check_desc.filterMode == tex_obj_wrapper.tex_desc.filterMode);
  REQUIRE(check_desc.flags == tex_obj_wrapper.tex_desc.flags);
}


TEST_CASE("Unit_hipGetTexObjectTextureDesc_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

  TexObjectTestWrapper tex_obj_wrapper(false);

  HIP_TEXTURE_DESC check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipTexObjectGetTextureDesc(nullptr, tex_obj_wrapper.texture_object),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipTexObjectGetTextureDesc(&check_desc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }
}

/* hipTexObjectDestroy test */

TEST_CASE("Unit_hipTexObjectDestroy_positive") {
  CHECK_IMAGE_SUPPORT;

  TexObjectTestWrapper tex_obj_wrapper(false, true);
  REQUIRE(hipTexObjectDestroy(tex_obj_wrapper.texture_object) == hipSuccess);
}
