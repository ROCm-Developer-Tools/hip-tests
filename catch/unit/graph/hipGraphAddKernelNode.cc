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
#include <hip_test_kernels.hh>

/**
 * @addtogroup hipGraphAddKernelNode hipGraphAddKernelNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
 * const hipGraphNode_t* pDependencies, size_t numDependencies,
 * const hipKernelNodeParams* pNodeParams)` -
 * Creates a kernel execution node and adds it to a graph.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipGraph_BasicFunctional
 */

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
 *    -# When the list of dependcencies is valid but the number of dependencies is not valid
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pointer to node params is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node params function data member is `nullptr`
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidDeviceFunction`
 *    -# When node params kernel params data member is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When adding kernel node to graph after graph is destroyed
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddKernelNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddKernelNode_Negative") {
  constexpr int N = 1024;
  size_t NElem{N};
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  int *A_d, *B_d, *C_d;
  hipGraph_t graph;
  hipGraphNode_t kNode;
  hipKernelNodeParams kNodeParams{};
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipMalloc(&A_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&B_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&C_d, sizeof(int) * N));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);

  SECTION("Pass pGraphNode as nullptr") {
    HIP_CHECK_ERROR(hipGraphAddKernelNode(nullptr, graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass Graph as nullptr") {
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, nullptr, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, nullptr, 11, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));
    dependencies.push_back(kNode);
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, dependencies.data(),
                                          dependencies.size() + 1, &kNodeParams),
                    hipErrorInvalidValue);
  }

  SECTION("Pass NodeParams as nullptr") {
    HIP_CHECK_ERROR(
        hipGraphAddKernelNode(&kNode, graph, dependencies.data(), dependencies.size(), nullptr),
        hipErrorInvalidValue);
  }

#if HT_NVIDIA  // on AMD this returns hipErrorInvalidValue
  SECTION("Pass NodeParams func data member as nullptr") {
    kNodeParams.func = nullptr;
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidDeviceFunction);
  }
#endif

  SECTION("Pass kernelParams data member as nullptr") {
    kNodeParams.kernelParams = nullptr;
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }

#if HT_AMD  // On Cuda setup this test case getting failed
  SECTION("Try adding kernel node after destroy the already created graph") {
    hipGraph_t destroyed_graph;
    HIP_CHECK(hipGraphCreate(&destroyed_graph, 0));
    HIP_CHECK(hipGraphDestroy(destroyed_graph));
    HIP_CHECK_ERROR(hipGraphAddKernelNode(&kNode, destroyed_graph, nullptr, 0, &kNodeParams),
                    hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipGraphDestroy(graph));
}
