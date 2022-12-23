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
 * @addtogroup hipGraphAddHostNode hipGraphAddHostNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
 * const hipGraphNode_t* pDependencies, size_t numDependencies,
 * const hipHostNodeParams* pNodeParams)` -
 * Creates a host execution node and adds it to a graph.
 */

#define SIZE 1024

static int* B_h;
static int* D_h;

static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < SIZE; i++) {
    A[i] = i;
  }
}

static void __global__ vector_square(int* B_d, int* D_d) {
  for (int i = 0; i < SIZE; i++) {
    D_d[i] = B_d[i] * B_d[i];
  }
}
static void vectorsquare_callback(void* ptr) {
  // The callback func is hipHostFn_t which is
  // of type void (*)(void*). This test is designed to
  // work with global variables, hence the workaround to
  // print this *ptr value to avoid type mismatch errors.
  int* A = reinterpret_cast<int*>(ptr);

  for (int i = 0; i < SIZE; i++) {
    if (D_h[i] != B_h[i] * B_h[i]) {
      INFO("Validation failed " << D_h[i] << B_h[i]);
      INFO("Ignore this garbage value" << *A);
      REQUIRE(false);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the graph node is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When number of dependencies is not valid
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When list of dependencies is valid but number of dependencies is not valid
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pointer to host params is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When host params host function data member is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddHostNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddHostNode_Negative") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  std::vector<hipGraphNode_t> dependencies;
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(hipGraphAddHostNode(nullptr, graph, nullptr, 0, &hostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    HIP_CHECK_ERROR(hipGraphAddHostNode(&hostNode, nullptr, nullptr, 0, &hostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(hipGraphAddHostNode(&hostNode, graph, nullptr, 11, &hostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
    dependencies.push_back(hostNode);
    HIP_CHECK_ERROR(hipGraphAddHostNode(&hostNode, graph, dependencies.data(),
                                        dependencies.size() + 1, &hostParams),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to host params") {
    HIP_CHECK_ERROR(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to host func") {
    hostParams.fn = nullptr;
    HIP_CHECK_ERROR(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams),
                    hipErrorInvalidValue);
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies API behaviour in cloned graph.
 *  - Creates graph and adds nodes.
 *  - Clones graph and adds host node to the cloned graph.
 *  - Validates the result.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddHostNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddHostNode_ClonedGraphWithHostNode") {
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

  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (A_h[i] != static_cast<int>(i)) {
      INFO("Validation failed i " << i << "C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This test case verifies the square of number by
creating graph, Add kernel node which does the square
of number and the result is validated by hipGraphAddHostNode API
*/
/**
 * Test Description
 * ------------------------
 *  - Verifies square of number by creating graph.
 *  - Adds kernel node which does the square of number.
 *  - Result is validated.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddHostNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddHostNode_VectorSquare") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *A_h{nullptr}, *B_d{nullptr}, *D_d{nullptr};
  int* param = reinterpret_cast<int*>(sizeof(int));

  HipTest::initArrays<int>(&A_d, &B_d, &D_d, &A_h, &B_h, &D_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_B, memcpyH2D_D, memcpyD2H_D, kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = vectorsquare_callback;
  hostParams.userData = param;

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_D, graph, nullptr, 0, D_d, D_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgs2[] = {&B_d, &D_d};
  kernelNodeParams.func = reinterpret_cast<void*>(vector_square);
  kernelNodeParams.gridDim = dim3(1);
  kernelNodeParams.blockDim = dim3(1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_D, graph, nullptr, 0, D_h, D_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_D, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_D, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_D, &hostNode, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HipTest::freeArrays<int>(A_d, B_d, D_d, A_h, B_h, D_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This test case verifies the following scenario
Create graph, calls the host function and updates
the parameters in the callback function and
validates it.
*/
/**
 * Test Description
 * ------------------------
 *  - Creates graph and calls the host function.
 *  - Updates the prameters in the callback function.
 *  - Validates results.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddHostNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddHostNode_BasicFunc") {
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

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (A_h[i] != static_cast<int>(i)) {
      INFO("Validation failed i " << i << "A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
