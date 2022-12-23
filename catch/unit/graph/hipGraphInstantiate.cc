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

/**
 * @addtogroup hipGraphInstantiate hipGraphInstantiate
 * @{
 * @ingroup GraphTest
 * `hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
 * hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize)` -
 * Creates an executable graph from a graph.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipGraph_BasicFunctional
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the executable graph is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph is not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When executable graph is not initialized
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphInstantiate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphInstantiate_Negative") {
  hipError_t ret;
  hipGraphExec_t gExec{};
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("Pass pGraphExec as nullptr") {
    ret = hipGraphInstantiate(nullptr, graph, nullptr, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graph as null/invalid ptr") {
    ret = hipGraphInstantiate(&gExec, nullptr, nullptr, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Graph as un-initialize") {
    hipGraph_t graph_uninit{};
    ret = hipGraphInstantiate(&gExec, graph_uninit, nullptr, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pGraphExec as un-initialize") {
    ret = hipGraphInstantiate(&gExec, graph, nullptr, nullptr, 0);
    REQUIRE(hipSuccess == ret);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Creates a graph.
 *  - Instantiates an executable graph.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphInstantiate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphInstantiate_Basic") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  REQUIRE(nullptr != graph);
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}
