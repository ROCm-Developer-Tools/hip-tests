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

#include <functional>

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphAddEventRecordNode hipGraphAddEventRecordNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode,
 * hipGraph_t graph, const hipGraphNode_t* pDependencies,
 * size_t numDependencies, hipEvent_t event)` -
 * Creates an event record node and adds it to a graph.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates a simple graph with just one event record node.
 *  - Instantiates the graph and launches it without errors.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_Simple") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0, event));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  // Wait for event
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

// Local test function
static void validateAddEventRecordNode(bool measureTime, bool withFlags, int nstep,
                                       unsigned flag = 0) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t ker_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, graph, nullptr, 0, &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::memsetReverse<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, graph, nullptr, 0, &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&ker_vecAdd, graph, nullptr, 0, &kernelNodeParams));
  hipEvent_t eventstart, eventend;
  if (withFlags) {
    HIP_CHECK(hipEventCreateWithFlags(&eventstart, flag));
    HIP_CHECK(hipEventCreateWithFlags(&eventend, flag));
  } else {
    HIP_CHECK(hipEventCreate(&eventstart));
    HIP_CHECK(hipEventCreate(&eventend));
  }
  hipGraphNode_t event_start, event_final;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_start, graph, nullptr, 0, eventstart));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_final, graph, nullptr, 0, eventend));
  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memset_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &ker_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &ker_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &ker_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &ker_vecAdd, &memcpyD2H_C, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_C, &event_final, 1));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int istep = 0; istep < nstep; istep++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    // Wait for eventend
    HIP_CHECK(hipEventSynchronize(eventend));
    // Verify graph execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, N);
    if (measureTime) {
      // Verify event record time difference_type
      float t = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&t, eventstart, eventend));
      REQUIRE(t > 0.0f);
    }
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(eventstart));
  HIP_CHECK(hipEventDestroy(eventend));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *  - Add different kinds of nodes to graph and add dependencies to nodes.
 *  - Create an event record node at the end.
 *  - Instantiate and launch the graph.
 *  - Wait for the event to complete.
 *  - Verify the results.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_WithoutFlags") {
  // Create events without flags using hipEventCreate and
  // elapsed time is not validated
  validateAddEventRecordNode(false, false, 1);
}

/**
 * Test Description
 * ------------------------
 *  - Add different kinds of nodes to graph and add dependencies to nodes.
 *  - Create event record nodes at the beginning and end.
 *  - Instantiate and launch the graph.
 *  - Wait for the event to complete.
 *  - Verify the results.
 *  - Also verify the elapsed time.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_ElapsedTime") {
  // Create events without flags using hipEventCreate and
  // elapsed time is validated
  validateAddEventRecordNode(true, false, 1);
}

/**
 * Test Description
 * ------------------------
 *  - Add different kinds of nodes to graph and add dependencies to nodes.
 *  - Create an event record nodes with flags at the end.
 *    -# When flag is `hipEventDefault`
 *    -# When flag is `hipEventBlockingSync`
 *    -# When flag is `hipEventDisableTiming`
 *  - Instantiate and launch graph.
 *  - Wait for the event to complete.
 *  - Verify the results.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_WithFlags") {
  // Create events with different flags using hipEventCreate and
  // elapsed time is not validated
  SECTION("Flag = hipEventDefault") {
    validateAddEventRecordNode(false, true, 1, hipEventDefault);
  }

  SECTION("Flag = hipEventBlockingSync") {
    validateAddEventRecordNode(false, true, 1, hipEventBlockingSync);
  }

  SECTION("Flag = hipEventDisableTiming") {
    validateAddEventRecordNode(false, true, 1, hipEventDisableTiming);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validate scenario @ref Unit_hipGraphAddEventRecordNode_Functional_WithoutFlags
 *    by running the graph multiple times in a loop
 * (100 times) after instantiation.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_MultipleRun") {
  validateAddEventRecordNode(false, false, 100);
}

/**
 * Test Description
 * ------------------------
 *  - Create event record node at the beginning with flag `hipEventDisableTiming`.
 *  - Add a memset node and event record nodes at the end.
 *  - Instantiate and launch the graph.
 *  - Wait for the event to complete.
 *  - Verify that elapsed time returns error.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_TimingDisabled") {
  constexpr size_t Nbytes = 1024;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));
  // memset node
  char* A_d;
  hipGraphNode_t memset_A;
  hipMemsetParams memsetParams{};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));

  hipGraphNode_t event_node_start, event_node_end;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_start, graph, nullptr, 0, event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_end, graph, nullptr, 0, event_end));
  // Add dependencies between nodes
  HIP_CHECK(hipGraphAddDependencies(graph, &event_node_start, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &event_node_end, 1));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  // Wait for event
  HIP_CHECK(hipEventSynchronize(event_end));
  // Validate hipEventElapsedTime returns error code because timing is
  // disabled for start and end event nodes.
  float t;
  HIP_CHECK_ERROR(hipEventElapsedTime(&t, event_start, event_end), hipErrorInvalidHandle);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_end));
  HIP_CHECK(hipEventDestroy(event_start));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *  - Validate several positive scenarios:
 *    -# When number of dependencies is zero, and dependencies are `nullptr`
 *      - Expected output: returned number for dependencies count is equal to 0
 *    -# When number of dependencies is less than total lenght
 *      - Expected output: returned number of dependencies count equal to 1
 *    -# When number of dependencies is equal to the total length
 *      - Expected output: returned number of depencencies is equal to the total length
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Positive_Parameters") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventrec;

  hipGraphNode_t dep_node = nullptr;
  hipGraphNode_t dep_node2 = nullptr;
  HIP_CHECK(hipGraphAddEmptyNode(&dep_node, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&dep_node2, graph, nullptr, 0));
  hipGraphNode_t dep_nodes[] = {dep_node, dep_node2};

  size_t numDeps = 0;
  SECTION("numDependencies is zero, dependencies is not nullptr") {
    HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, dep_nodes, 0, event));
    HIP_CHECK(hipGraphNodeGetDependencies(eventrec, nullptr, &numDeps));
    REQUIRE(numDeps == 0);
  }

  SECTION("numDependencies < dependencies length") {
    HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, dep_nodes, 1, event));
    HIP_CHECK(hipGraphNodeGetDependencies(eventrec, nullptr, &numDeps));
    REQUIRE(numDeps == 1);
  }

  SECTION("numDependencies == dependencies length") {
    HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, dep_nodes, 2, event));
    HIP_CHECK(hipGraphNodeGetDependencies(eventrec, nullptr, &numDeps));
    REQUIRE(numDeps == 2);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When graph handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node dependencies are `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When dependencies are not `nullptr` and the size is not zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node in dependency is from different graph
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When number of nodes is not valid (0)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When duplicate node in dependencies
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node event handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph is not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When event is not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddEventRecordNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Negative") {
  using namespace std::placeholders;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventrec;

  GraphAddNodeCommonNegativeTests(std::bind(hipGraphAddEventRecordNode, _1, _2, _3, _4, event),
                                  graph);

  SECTION("event = nullptr") {
    HIP_CHECK_ERROR(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    HIP_CHECK_ERROR(hipGraphAddEventRecordNode(&eventrec, graph_uninit, nullptr, 0, event),
                    hipErrorInvalidValue);
  }

  SECTION("event is uninitialized") {
    hipEvent_t event_uninit{};
    HIP_CHECK_ERROR(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0, event_uninit),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}
