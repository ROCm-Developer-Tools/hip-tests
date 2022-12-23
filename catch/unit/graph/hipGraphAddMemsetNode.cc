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

#include <hip_test_defgroups.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphAddMemsetNode hipGraphAddMemsetNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemsetNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, const hipMemsetParams *pMemsetParams)` -
 * Creates a memset node and adds it to a graph.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipGraph_BasicFunctional
 */

/**
 * Test Description
 * ------------------------
 *  - Verify that all elements of destination memory are set to the correct value.
 *  - The test is repeated for all valid element sizes(1, 2, 4), and several allocations of different
 *    height and width.
 *  - The test is repeated both on host and device.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipGraphAddMemsetNode_Positive_Basic", "", uint8_t, uint16_t, uint32_t) {
  const auto f = [](hipMemsetParams* params) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, params));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  GraphMemsetNodeCommonPositive<TestType>(f);
}

/**
 * Test Description
 * ------------------------
 *  - Verify API behaviour with invalid arguments:
 *    -# When pGraphNode is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pDependencies is `nullptr` and numDependencies is not zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When a node in pDependencies originates from a different graph
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When numNodes is invalid
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When a node is duplicated in pDependencies
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams dst data member is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams elementSize data member is different from 1, 2, and 4
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams width data member is zero
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams width data member is larger than the allocated memory region
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams height data member is zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams pitch data memebr is less than width when height is more than 1
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pMemsetParams pitch * pMemsetParams height is larger than the allocated memory region
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddMemsetNode_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, 4 * sizeof(int));
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(*alloc.ptr());
  params.width = 1;
  params.height = 1;
  params.value = 42;

  GraphAddNodeCommonNegativeTests(std::bind(hipGraphAddMemsetNode, _1, _2, _3, _4, &params), graph);

  hipGraphNode_t node = nullptr;
  MemsetCommonNegative(std::bind(hipGraphAddMemsetNode, &node, graph, nullptr, 0, _1), params);

  HIP_CHECK(hipGraphDestroy(graph));
}
