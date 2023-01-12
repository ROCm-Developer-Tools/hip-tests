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
 * @addtogroup hipGraphExecDestroy hipGraphExecDestroy
 * @{
 * @ingroup GraphTest
 * `hipGraphExecDestroy(hipGraphExec_t graphExec)` -
 * Destroys an executable graph
 */

static void HostFunctionSetToZero(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) = 0;
}

static void HostFunctionAddOne(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) += 1;
}

/* create an executable graph that will set an integer pointed to by 'number' to one*/
static void CreateTestExecutableGraph(hipGraphExec_t* graph_exec, int* number) {
  hipGraph_t graph;
  hipGraphNode_t node_error;

  hipGraphNode_t node_set_zero;
  hipHostNodeParams params_set_to_zero = {HostFunctionSetToZero, number};

  hipGraphNode_t node_add_one;
  hipHostNodeParams params_set_add_one = {HostFunctionAddOne, number};

  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddHostNode(&node_set_zero, graph, nullptr, 0, &params_set_to_zero));
  HIP_CHECK(hipGraphAddHostNode(&node_add_one, graph, &node_set_zero, 1, &params_set_add_one));

  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, &node_error, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphExecDestroy
 *    - create an executable graph and then destroy it
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecDestroy_Positive_Basic") {
  int number = 5;
  hipGraphExec_t graph_exec;
  CreateTestExecutableGraph(&graph_exec, &number);
  REQUIRE(hipGraphExecDestroy(graph_exec) == hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *    - Basic negative parameter test for hipGraphExecDestroy
 *    - try to destroy an empty hipGraphExec_t object
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecDestroy_Negative_Parameters") {
  hipGraphExec_t graph_exec{};
  HIP_CHECK_ERROR(hipGraphExecDestroy(graph_exec), hipErrorInvalidValue);
}
