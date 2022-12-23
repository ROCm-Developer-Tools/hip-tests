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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

/**
 * @addtogroup hipGraphExecHostNodeSetParams hipGraphExecHostNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
 * const hipHostNodeParams* pNodeParams)` -
 * Sets the parameters for a host node in the given graphExec.
 */

#define SIZE 1024

void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }
}

void callbackfunc_setparams(void* B_h) {
  int* B = reinterpret_cast<int*>(B_h);
  for (int i = 0; i < SIZE; i++) {
    B[i] = i * i;
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When graph exec handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When host node params pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When host params function data member is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When host params are not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node is not a host node
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node is not instantiated
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecHostNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecHostNodeSetParams_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyD2H_AC, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr, 0, A_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_AC, &hostNode, 1));

  hipHostNodeParams sethostParams = {0, 0};
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;

  hipGraphNode_t empty_node;
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, &hostNode, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Passing nullptr to graphExec") {
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(nullptr, hostNode, &sethostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to hostParams") {
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(graphExec, hostNode, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(graphExec, nullptr, &sethostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to host func") {
    sethostParams.fn = nullptr;
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(graphExec, hostNode, &sethostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Passing uninitialized hostParams") {
    hipHostNodeParams unintParams = {0, 0};
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(graphExec, hostNode, &unintParams),
                    hipErrorInvalidValue);
  }

#if HT_NVIDIA // segfaults on AMD
  SECTION("node is not a host node") {
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(graphExec, empty_node, &sethostParams),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("node is not instantiated") {
    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
    HIP_CHECK_ERROR(hipGraphExecHostNodeSetParams(graphExec, hostNode, &sethostParams),
                    hipErrorInvalidValue);
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies API behaviour in a cloned graph.
 *  - Creates graph and adds graph nodes.
 *  - Clones the graph and adds host node to the cloned graph.
 *  - Update the host params using set API.
 *  - Validates the result.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecHostNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecHostNodeSetParams_ClonedGraphWithHostNode") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_C, memcpyD2H_AC;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr, 0, A_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  hipGraph_t clonedgraph;
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, clonedgraph, nullptr, 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C, &memcpyD2H_AC, 1));

  hipHostNodeParams sethostParams = {0, 0};
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;

  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecHostNodeSetParams(graphExec, hostNode, &sethostParams));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != static_cast<int>(i * i)) {
      INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *  - Creates graph and adds host node to the graph.
 *  - Updates the host params using set functionality.
 *  - Validates the result.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecHostNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecHostNodeSetParams_BasicFunc") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyD2H_AC, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr, 0, A_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_AC, &hostNode, 1));

  hipHostNodeParams sethostParams = {0, 0};
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecHostNodeSetParams(graphExec, hostNode, &sethostParams));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != static_cast<int>(i * i)) {
      INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
