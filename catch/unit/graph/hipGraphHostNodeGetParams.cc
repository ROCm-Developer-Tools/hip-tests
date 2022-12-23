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
 * @addtogroup hipGraphHostNodeGetParams hipGraphHostNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams)` -
 * Returns a host node's parameters.
 */

#define SIZE 1024

static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }
}

static void callbackfunc_setparams(void* B_h) {
  int* B = reinterpret_cast<int*>(B_h);
  for (int i = 0; i < SIZE; i++) {
    B[i] = i * i;
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When graph node handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When output pointer to the node params is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node is not a host node
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphHostNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphHostNodeGetParams_Negative") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  hipHostNodeParams GethostParams;

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(hipGraphHostNodeGetParams(nullptr, &GethostParams), hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to hostParams") {
    HIP_CHECK_ERROR(hipGraphHostNodeGetParams(hostNode, nullptr), hipErrorInvalidValue);
  }

#if HT_NVIDIA // segfaults on AMD
  SECTION("node is not a host node") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphHostNodeGetParams(empty_node, &GethostParams), hipErrorInvalidValue);
  }
#endif

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies API behaviour in cloned graph.
 *  - Creates graph and adds graph nodes.
 *  - Clones the graph and adds host node to the cloned graph.
 *  - Updates host node by setting its params.
 *  - Gets the host node params and compare them with the ones that are set.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphHostNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphHostNodeGetParams_ClonedGraphWithHostNode") {
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
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C, &memcpyD2H_AC, 1));

  hipGraph_t clonedgraph;
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, clonedgraph, nullptr, 0, &hostParams));

  hipHostNodeParams sethostParams = {0, 0};
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;
  HIP_CHECK(hipGraphHostNodeSetParams(hostNode, &sethostParams));
  hipHostNodeParams gethostParams;
  HIP_CHECK(hipGraphHostNodeGetParams(hostNode, &gethostParams));
  REQUIRE(memcmp(&sethostParams, &gethostParams, sizeof(hipHostNodeParams)) == 0);

  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != static_cast<int>(i * i)) {
      INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
}

/*
This test case verifies the following scenarios
Create graph, Adds host node to the graph, updates it
with hipGraphHostNodeSetParams and gets the host node
params using hipGraphHostNodeGetParams API and validates
it
*/
void hipGraphHostNodeGetParams_func(bool setparams) {
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

  if (setparams) {
    hipHostNodeParams sethostParams = {0, 0};
    sethostParams.fn = callbackfunc_setparams;
    sethostParams.userData = C_h;
    HIP_CHECK(hipGraphHostNodeSetParams(hostNode, &sethostParams));

    hipHostNodeParams gethostParams;
    HIP_CHECK(hipGraphHostNodeGetParams(hostNode, &gethostParams));
    REQUIRE(memcmp(&sethostParams, &gethostParams, sizeof(hipHostNodeParams)) == 0);
  } else {
    hipHostNodeParams gethostParams;
    HIP_CHECK(hipGraphHostNodeGetParams(hostNode, &gethostParams));
    REQUIRE(memcmp(&hostParams, &gethostParams, sizeof(hipHostNodeParams)) == 0);
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  if (setparams) {
    for (size_t i = 0; i < N; i++) {
      if (C_h[i] != static_cast<int>(i * i)) {
        INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
        REQUIRE(false);
      }
    }
  } else {
    for (size_t i = 0; i < N; i++) {
      if (A_h[i] != static_cast<int>(i)) {
        INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
        REQUIRE(false);
      }
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
 *  - Adds host node to the graph.
 *  - Gets host params and validates them.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphHostNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphHostNodeGetParams_BasicFunc") { hipGraphHostNodeGetParams_func(false); }

/*
This test case verifies hipGraphHostNodeGetParams API by
adding host node to graph, updates host node params
using hipGraphHostNodeSetParams  and gets the host params
validates it
*/
/**
 * Test Description
 * ------------------------
 *  - Adds host node to the graph.
 *  - Updates host node params using set function.
 *  - Gets params and validates them.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphHostNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphHostNodeGetParams_SetParams") { hipGraphHostNodeGetParams_func(true); }
