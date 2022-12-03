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

#include <functional>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <resource_guards.hh>

#include "graph_memcpy_to_from_symbol_common.hh"

HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(char)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(int)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(float)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(double)

HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(char)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(int)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(float)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(double)

template <typename T>
void GraphExecMemcpyFromSymbolSetParamsShell(void* symbol, void* alt_symbol, size_t offset,
                                             const std::vector<T> expected) {
  const auto f = [alt_symbol, is_arr = expected.size() > 1](void* dst, void* symbol, size_t count,
                                                            size_t offset,
                                                            hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;

    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(
        &node, graph, nullptr, 0, reinterpret_cast<T*>(dst) + is_arr, alt_symbol,
        count - is_arr * sizeof(T), offset + is_arr * sizeof(T), direction));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(graph_exec, node, dst, symbol, count,
                                                        offset, direction));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  MemcpyFromSymbolShell(f, symbol, offset, std::move(expected));
}

#define HIP_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_TEST(type)                               \
  SECTION("Scalar variable") {                                                                     \
    GraphExecMemcpyFromSymbolSetParamsShell(HIP_SYMBOL(type##_device_var),                         \
                                            HIP_SYMBOL(type##_alt_device_var), 0,                  \
                                            std::vector<type>{5});                                 \
  }                                                                                                \
                                                                                                   \
  SECTION("Constant scalar variable") {                                                            \
    GraphExecMemcpyFromSymbolSetParamsShell(HIP_SYMBOL(type##_const_device_var),                   \
                                            HIP_SYMBOL(type##_alt_const_device_var), 0,            \
                                            std::vector<type>{5});                                 \
  }                                                                                                \
                                                                                                   \
  SECTION("Array") {                                                                               \
    const auto offset = GENERATE(0, kArraySize / 2);                                               \
    INFO("Array offset: " << offset);                                                              \
    std::vector<type> expected(kArraySize - offset);                                               \
    std::iota(expected.begin(), expected.end(), offset + 1);                                       \
    GraphExecMemcpyFromSymbolSetParamsShell(HIP_SYMBOL(type##_device_arr),                         \
                                            HIP_SYMBOL(type##_alt_device_arr), offset,             \
                                            std::move(expected));                                  \
  }                                                                                                \
                                                                                                   \
  SECTION("Constant array") {                                                                      \
    const auto offset = GENERATE(0, kArraySize / 2);                                               \
    INFO("Array offset: " << offset);                                                              \
    std::vector<type> expected(kArraySize - offset);                                               \
    std::iota(expected.begin(), expected.end(), offset + 1);                                       \
    GraphExecMemcpyFromSymbolSetParamsShell(HIP_SYMBOL(type##_const_device_arr),                   \
                                            HIP_SYMBOL(type##_alt_const_device_arr), offset,       \
                                            std::move(expected));                                  \
  }

TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Positive_Basic") {
  SECTION("char") { HIP_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_TEST(char); }

  SECTION("int") { HIP_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_TEST(int); }

  SECTION("float") { HIP_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_TEST(float); }

  SECTION("double") { HIP_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_TEST(double); }
}

TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> var(LinearAllocs::hipMalloc, sizeof(int));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&node, graph, nullptr, 0, var.ptr(),
                                            HIP_SYMBOL(int_device_var), sizeof(*var.ptr()), 0,
                                            hipMemcpyDefault));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  MemcpyFromSymbolCommonNegative(
      std::bind(hipGraphExecMemcpyNodeSetParamsFromSymbol, graph_exec, node, _1, _2, _3, _4, _5),
      var.ptr(), HIP_SYMBOL(int_device_var), sizeof(*var.ptr()));

  SECTION("Changing memcpy direction") {
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParamsFromSymbol(
                        graph_exec, node, var.ptr(), HIP_SYMBOL(int_device_var), sizeof(*var.ptr()),
                        0, hipMemcpyDeviceToHost),
                    hipErrorInvalidValue);
  }

  SECTION("Changing dst allocation device") {
    if (HipTest::getDeviceCount() < 2) {
      HipTest::HIP_SKIP_TEST("Test requires two connected GPUs");
      return;
    }
    HIP_CHECK(hipSetDevice(1));
    LinearAllocGuard<int> new_var(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParamsFromSymbol(
                        graph_exec, node, new_var.ptr(), HIP_SYMBOL(int_device_var),
                        sizeof(*new_var.ptr()), 0, static_cast<hipMemcpyKind>(-1)),
                    hipErrorInvalidMemcpyDirection);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}