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

/**
Testcase Scenarios
------------------
Functional ::
 1) Add nodes to graph with and without dependencies, verify the api returns list of
 root nodes (i.e., nodes without dependencies).
 2) Pass nodes as nullptr and verify api returns actual number of root nodes added to graph.
 3) If NumRootNodes passed is greater than the actual number of root nodes, the remaining entries in
 nodes list will be set to NULL, and the number of nodes actually obtained will be returned in NumRootNodes.
 4) Create a graph with stream capture done on multiple dependent streams.
 Verify root nodes of created graph are matching the operations pushed which doesn't have dependencies.

Argument Validation ::
 1) Pass graph as nullptr and verify api returns error code.
 2) Pass numRootNodes as nullptr and other params as valid values. Expect api to return error code.
 3) When there are no nodes in graph, expect numRootNodes to be set to zero.
 4) Pass numRootNodes less than actual number of nodes. Expect api to populate requested number of node entries
 and does update numRootNodes.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

namespace {
inline constexpr size_t kNumOfRootNodes = 2;
}  // anonymous namespace

/**
 * Local Function to validate number of root nodes.
 */
static void validate_hipGraphGetRootNodes(size_t numNodesToGet,
                                        int testnum,
                                        std::vector<hipGraphNode_t> nodelist,
                                        hipGraph_t graph) {
  size_t numNodes = numNodesToGet;
  hipGraphNode_t *nodes = new hipGraphNode_t[numNodes]{};
  int found_count = 0;
  HIP_CHECK(hipGraphGetRootNodes(graph, nodes, &numNodes));
  // Verify added nodes are present in the node entries returned
  for (size_t i = 0; i < kNumOfRootNodes; i++) {
    for (size_t j = 0; j < numNodes; j++) {
      if (nodelist[i] == nodes[j]) {
        found_count++;
        break;
      }
    }
  }
  // Validate
  if (testnum == 0) {
    REQUIRE(found_count == kNumOfRootNodes);
  } else if (testnum == 1) {
    // Verify numNodes is unchanged
    REQUIRE(numNodes == numNodesToGet);
    REQUIRE(found_count == numNodesToGet);
  } else if (testnum == 2) {
    // Verify numNodes is reset to actual number of nodes
    REQUIRE(numNodes == nodelist.size());
    REQUIRE(found_count == kNumOfRootNodes);
    // Verify additional entries in nodes are set to nullptr
    for (auto i = numNodes; i < numNodesToGet; i++) {
      REQUIRE(nodes[i] == nullptr);
    }
  }

  delete[] nodes;
}

/**
 * Functional Test for API fetching root node list
 */
TEST_CASE("Unit_hipGraphGetRootNodes_Positive_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;

  hipGraphNode_t memcpyNode, kernelNode;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies, rootnodelist;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  rootnodelist.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  rootnodelist.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, dependencies.data(),
                                  dependencies.size(), &kernelNodeParams));
  dependencies.clear();
  dependencies.push_back(kernelNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(), C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  size_t numRootNodes{};
  // Get numRootNodes by passing rootnodes list as nullptr.
  // verify : numRootNodes is set to actual number of root nodes added
  // Scenario 1
  SECTION("Validate number of rootnodes") {
    HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
    INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);
    REQUIRE(numRootNodes == rootnodelist.size());
  }

  // Scenario 2
  SECTION("Validate root node list when numRootNodes = num of nodes") {
    validate_hipGraphGetRootNodes(kNumOfRootNodes, 0,
                                    rootnodelist, graph);
  }

  // Scenario 3
  SECTION("Validate root node list when numRootNodes < num of nodes") {
    validate_hipGraphGetRootNodes(kNumOfRootNodes - 1, 1,
                                    rootnodelist, graph);
  }

  // Scenario 4
  SECTION("Validate root node list when numRootNodes > num of nodes") {
    validate_hipGraphGetRootNodes(kNumOfRootNodes + 1, 2,
                                    rootnodelist, graph);
  }

  // Scenario 5
  SECTION("Validate numRootNodes is 0 when no nodes in graph") {
    hipGraph_t emptyGraph{};
    HIP_CHECK(hipGraphCreate(&emptyGraph, 0));
    HIP_CHECK(hipGraphGetRootNodes(emptyGraph, nullptr, &numRootNodes));
    REQUIRE(numRootNodes == 0);
    HIP_CHECK(hipGraphDestroy(emptyGraph));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}


/**
 * Create a graph with stream capture done on multiple dependent streams. Verify root nodes
 * of created graph are matching the operations pushed which doesn't have dependencies.
 */
TEST_CASE("Unit_hipGraphGetRootNodes_CapturedStream") {
  hipStream_t stream1{nullptr}, stream2{nullptr}, mstream{nullptr};
  hipStream_t streamForGraph{nullptr};
  hipEvent_t memsetEvent1, memsetEvent2, forkStreamEvent;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 1000000;
  constexpr int numMemsetNodes = 2;
  size_t Nbytes = N * sizeof(float), numRootNodes{};
  float *A_d, *C_d;
  float *A_h, *C_h;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  HIP_CHECK(hipStreamCreate(&streamForGraph));

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
      A_h[i] = 3.146f + i;  // Pi
  }

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&mstream));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipStreamBeginCapture(mstream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, mstream));
  HIP_CHECK(hipStreamWaitEvent(stream1, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipEventRecord(memsetEvent1, stream1));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent1, 0));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent2, 0));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mstream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, mstream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mstream));
  HIP_CHECK(hipStreamEndCapture(mstream, &graph));

  // Verify numof root nodes
  HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
  REQUIRE(numRootNodes == numMemsetNodes);
  INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);

  int numBytes = sizeof(hipGraphNode_t) * numRootNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(nodes != nullptr);

  hipGraphNodeType nodeType;
  HIP_CHECK(hipGraphGetRootNodes(graph, nodes, &numRootNodes));
  REQUIRE(numRootNodes == numMemsetNodes);

  // Verify root nodes returned are memset nodes.
  HIP_CHECK(hipGraphNodeGetType(nodes[0], &nodeType));
  REQUIRE(nodeType == hipGraphNodeTypeMemset);
  HIP_CHECK(hipGraphNodeGetType(nodes[1], &nodeType));
  REQUIRE(nodeType == hipGraphNodeTypeMemset);

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      INFO("A and C not matching at " << i << " C_h[i] " << C_h[i]
                                           << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipStreamDestroy(mstream));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memsetEvent1));
  HIP_CHECK(hipEventDestroy(memsetEvent2));
  free(A_h);
  free(C_h);
  free(nodes);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}


/**
 * Test performs api parameter validation by passing various values
 * as input and output parameters and validates the behavior.
 * Test will include both negative and positive scenarios.
 */
TEST_CASE("Unit_hipGraphGetRootNodes_ParamValidation") {
  hipStream_t stream1{nullptr}, stream2{nullptr}, mstream{nullptr};
  hipEvent_t memsetEvent1, memsetEvent2, forkStreamEvent;
  hipGraph_t graph{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(float), numRootNodes{};
  float *A_d, *C_d;
  float *A_h, *C_h;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&mstream));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipStreamBeginCapture(mstream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, mstream));
  HIP_CHECK(hipStreamWaitEvent(stream1, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipEventRecord(memsetEvent1, stream1));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent1, 0));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent2, 0));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mstream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, mstream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mstream));
  HIP_CHECK(hipStreamEndCapture(mstream, &graph));
  HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
  INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);
  int numBytes = sizeof(hipGraphNode_t) * numRootNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(nodes != nullptr);

  SECTION("graph as nullptr") {
    HIP_CHECK_ERROR(hipGraphGetRootNodes(nullptr, nodes, &numRootNodes), hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    HIP_CHECK_ERROR(hipGraphGetRootNodes(graph_uninit, nodes, &numRootNodes), hipErrorInvalidValue);
  }

  SECTION("numRootNodes as nullptr") {
    HIP_CHECK_ERROR(hipGraphGetRootNodes(graph, nodes, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(mstream));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memsetEvent1));
  HIP_CHECK(hipEventDestroy(memsetEvent2));
  free(A_h);
  free(C_h);
  free(nodes);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
