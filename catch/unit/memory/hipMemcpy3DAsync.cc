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

#include <memcpy1d_tests_common.hh>
#include <memcpy3d_tests_common.hh>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpy3DAsync hipMemcpy3DAsync
 * @{
 * @ingroup MemoryTest
 * `hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream __dparm(0))` -
 * Copies data between host and device asynchronously.
 */

/**
 * Test Description
 * ------------------------
 *  - Verifies basic test cases for copying 3D memory between
 *    device and host asynchronously.
 *  - Validates following memcpy directions:
 *    -# Device to host
 *    -# Device to device
 *      - Peer access disabled
 *      - Peer access enabled
 *    -# Host to device
 *    -# Host to host
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Basic") {
  constexpr bool async = true;

  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  SECTION("Device to Host") { Memcpy3DDeviceToHostShell<async>(Memcpy3DWrapper<async>, stream); }

  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(Memcpy3DWrapper<async>, stream);
    }
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(Memcpy3DWrapper<async>, stream);
    }
  }

  SECTION("Host to Device") { Memcpy3DHostToDeviceShell<async>(Memcpy3DWrapper<async>, stream); }

  SECTION("Host to Host") { Memcpy3DHostToHostShell<async>(Memcpy3DWrapper<async>, stream); }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that API synchronizes regarding to host when copying from
 *    device memory to the pageable or pinned host memory.
 *  - Validates following memcpy directions:
 *    -# Host to device
 *    -# Device to pageable host
 *      - Platform specific (NVIDIA)
 *    -# Device to pinned host
 *    -# Device to device
 *    -# Host to host
 *      - Platform specific (NVIDIA)
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Synchronization_Behavior") {
  constexpr bool async = true;

  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy3DHtoDSyncBehavior(Memcpy3DWrapper<async>, false); }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-233
  SECTION("Device to Pageable Host") {
    Memcpy3DDtoHPageableSyncBehavior(Memcpy3DWrapper<async>, true);
  }
#endif

  SECTION("Device to Pinned Host") {
    Memcpy3DDtoHPinnedSyncBehavior(Memcpy3DWrapper<async>, false);
  }

  SECTION("Device to Device") { Memcpy3DDtoDSyncBehavior(Memcpy3DWrapper<async>, false); }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-233
  SECTION("Host to Host") { Memcpy3DHtoHSyncBehavior(Memcpy3DWrapper<async>, true); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates that nothing will be copied if width, height or depth are set to zero.
 *  - Validates following memcpy directions:
 *    -# Device to host
 *    -# Device to device
 *    -# Host to device
 *    -# Host to host
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Parameters") {
  constexpr bool async = true;
  Memcpy3DZeroWidthHeightDepth<async>(Memcpy3DWrapper<async>);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that an array is successfully copied.
 *  - Validates following memcpy directions:
 *    -# Device to host
 *    -# Host to host
 *    -# Device to device
 *      - Plaform specific (NVIDIA)
 *    -# Host to device
 *      - Plaform specific (NVIDIA)
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Array") {
  constexpr bool async = true;
  SECTION("Array from/to Host") { Memcpy3DArrayHostShell<async>(Memcpy3DWrapper<async>); }
#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-238
  SECTION("Array from/to Device") { Memcpy3DArrayDeviceShell<async>(Memcpy3DWrapper<async>); }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When destination pitch is less than width
 *      - Expected output: return `hipErrorInvalidPitchValue`
 *    -# When source pitch is less than width
 *      - Expected output: return `hipErrorInvalidPitchValue`
 *    -# When destination pitch is larger than maximum pitch
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When source pitch is larger than maximum pitch
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When extent.width + dst_pos.x > dst_ptr.pitch
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When extent.width + src_pos.x > src_ptr.pitch
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When dst_pos.y out of bounds
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When src_pos.y out of bounds
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When dst_pos.z out of bounds
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When src_pos.z out of bounds
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memcpy kind is not valid (-1)
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When stream is not valid
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorContextIsDestroyed`
 *  - All cases are executed for following memcpy directions:
 *    -# Host to device
 *    -# Device to host
 *    -# Host to host
 *    -# Device to device
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy3DAsync_Negative_Parameters") {
  constexpr bool async = true;
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-239
    SECTION("dst_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("src_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidPitchValue);
    }
#endif

    SECTION("dst_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-237
    SECTION("extent.width + dst_pos.x > dst_ptr.pitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("extent.width + src_pos.x > src_ptr.pitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.y out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.y out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.z out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.z out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }
#endif

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-234
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, src_pos, extent,
                                             static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidMemcpyDirection);
    }
#endif

#if HT_NVIDIA // Disabled on AMD due to defect - EXSWHTEC-235
    SECTION("Invalid stream") {
      StreamGuard stream_guard(Streams::created);
      HIP_CHECK(hipStreamDestroy(stream_guard.stream()));
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind,
                                             stream_guard.stream()),
                      hipErrorContextIsDestroyed);
    }
#endif
  };

  SECTION("Host to Device") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0), extent,
                  hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    NegativeTests(make_hipPitchedPtr(dst_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard3D<int> src_alloc(extent);
    LinearAllocGuard3D<int> dst_alloc(extent);
    NegativeTests(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                  make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice);
  }
}
