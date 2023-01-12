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

#include "hipAPICoverageUtils.h"

/*
Used to find all HIP API occurrences within the passed test .cc files.
Occurrence is detected when the HIP API is called in several scenarios.
    - API Call with assertion, e.g. REQUIRE(hipAPI());
    - API call with assignment, e.g. result = hipAPI();
    - API call with parameter, e.g. ASSERT_EQUAL(hipSuccess, hipGetDevice(&deviceId));
*/
void findAPICallInFile(HipAPI& hip_api, std::string test_module_file) {
  std::fstream test_module_file_handler;
  test_module_file_handler.open(test_module_file);

  int line_number{0};
  std::string line;
  /*
  Closed brackets might be in another line, if the API
  call is multiline.
  */
  std::string api_call_with_assert{"(" + hip_api.getName() + "("};
  std::string api_call_with_assignment{"= " + hip_api.getName() + "("};
  std::string api_call_with_parameter{", " + hip_api.getName() + "("};

  while (std::getline(test_module_file_handler, line)) {
    ++line_number;
    if ((line.find(api_call_with_assert) != std::string::npos) ||
        (line.find(api_call_with_assignment) != std::string::npos) ||
        (line.find(api_call_with_parameter) != std::string::npos)) {
      hip_api.addFileOccurrence(FileOccurrence(test_module_file, line_number));
    }
  }

  test_module_file_handler.close();
}

/*
Used to find all HIP API test cases within the passed test .cc files.
Matching test case is detected when the HIP API in defined within doxygen comment.
*/
void findAPITestCaseInFile(HipAPI& hip_api, std::string test_module_file) {
  std::fstream test_module_file_handler;
  test_module_file_handler.open(test_module_file);

  int line_number{0};
  std::string line;

  std::string add_group_definition{"@addtogroup"};
  std::string ref_test_case{"@ref"};
  std::string test_case_definition{"TEST_CASE("};
  std::string current_api_name{"None"};
  std::string test_case{"None"};

  while (std::getline(test_module_file_handler, line)) {
    ++line_number;
    if (line.find(add_group_definition) != std::string::npos) {
      current_api_name = line.substr(line.find(add_group_definition) + 1);
      current_api_name = current_api_name.substr(current_api_name.rfind(" ") + 1);
    }

    if (hip_api.getName() != current_api_name) {
      continue;
    }

    if (line.find(ref_test_case) != std::string::npos) {
      test_case = line.substr(line.rfind(" ") + 1);
      hip_api.addTestCase(TestCaseOccurrence{test_case, test_module_file, line_number});
      continue;
    }

    if (line.find(test_case_definition) != std::string::npos) {
      test_case = line.substr(line.find("\"") + 1);
      test_case = test_case.substr(0, test_case.find("\""));
      hip_api.addTestCase(TestCaseOccurrence{test_case, test_module_file, line_number});
    }
  }

  test_module_file_handler.close();
}

/*
Used to iterate through all passed test .cc files and search for passed
HIP API instance. This instance shall be used to update occurrences.
*/
void searchForAPI(HipAPI& hip_api, std::vector<std::string>& test_module_files) {
  std::cout << "Searching for " << hip_api.getName() << " in test module files." << std::endl;
  for (auto const& test_module_file: test_module_files) {
    findAPICallInFile(hip_api, test_module_file);
    findAPITestCaseInFile(hip_api, test_module_file);
  }
}

/*
Used to extract all HIP APIs from the passed header file.
*/
std::vector<HipAPI> extractHipAPIs(std::string& hip_api_header_file,
    std::vector<std::string>& api_group_names, bool start_groups) {
  std::fstream hip_header_file_handler;
  hip_header_file_handler.open(hip_api_header_file);

  std::string line;
  std::vector<HipAPI> hip_apis;

  /*
  If the HIP API is deprecated, it will be marked with
  the following deprecation line in code.
  */
  std::string deprecated_line{"DEPRECATED("};
  /*
  Each HIP API has prefix hip in the name. Groups are marked with @defgroup, and the
  main group that shall be considered is HIP API. Before that group is defined, lines
  of code shall not be considered.
  */
  std::string hip_api_prefix{"hip"};
  std::string group_definition{"@defgroup"};
  std::string add_group_definition{"@addtogroup"};
  std::string start_of_api_groups{"HIP API"};
  std::string end_of_api_groups{"doxygen end HIP API"};
  std::string end_group_definition{"@}"};

  bool deprecated_flag{false};
  bool api_group_names_start{start_groups};
  std::vector<std::string> api_group_names_tracker{api_group_names};
  int line_number{0};

  while (std::getline(hip_header_file_handler, line)) {
    ++line_number;

    // Declarations of the HIP APIs start after the HIP API group has been defined.
    if ((line.find(group_definition) != std::string::npos || line.find(add_group_definition) != std::string::npos)
        && line.find(start_of_api_groups) != std::string::npos) {
      api_group_names_start = true;
      continue;
    }

    // If the API Groups have not started yet, go to the next file line.
    if (!api_group_names_start) {
      continue;
    }

    // If the end of HIP API group has been detected, break the loop.
    if (line.find(end_of_api_groups) != std::string::npos) {
      break;
    }

    /*
    If the API is deprecated, raise a flag and go to the
    next line where the API is declared.
    */ 
    if (line.find(deprecated_line) != std::string::npos) {
      std::getline(hip_header_file_handler, line);
      ++line_number;
      deprecated_flag = true;
    } else {
      deprecated_flag = false;
    }

    if (line.find(group_definition) != std::string::npos) {
      std::string group_name = line.substr(line.find(group_definition) + group_definition.length() + 1);
      group_name = group_name.substr(group_name.find(' ') + 1);
      api_group_names.push_back(group_name);
      api_group_names_tracker.push_back(group_name);
    }
    else if (line.find(add_group_definition) != std::string::npos)
    {
      std::string group_name = line.substr(line.find(add_group_definition) + add_group_definition.length() + 1);
      group_name = group_name.substr(group_name.find(' ') + 1);
      api_group_names.push_back(group_name);
      api_group_names_tracker.push_back(group_name);
    }

    /*
    Possible case is that there are nested groups. While api_group_names
    vector contains all detected groups, api_group_names_tracker is responsible
    to track the last defined group, because of the nested cases.
    */
    if (line.find(end_group_definition) != std::string::npos) {
      api_group_names_tracker.pop_back();
    }

    /*
    The line which contains HIP API declaration looks like:
    hipError_t hipMalloc(void** ptr, size_t size);
    The name of HIP API is found by following steps:
        - Take the substring from the start to the first open bracket.
        - Extract the name from that substring by finding the last "hip".
    Avoid comments.
    */
    if (line.find(hip_api_prefix) != std::string::npos && 
        line.find("(") != std::string::npos &&
        line.find("  ") != 0 &&
        line.find(" *") != 0) {
      std::string api_name_no_brackets{line.substr(0, line.find("("))};
      /*
      If there is no hip substring, then there is no valid API in that line.
      */
      if (api_name_no_brackets.rfind(hip_api_prefix) == std::string::npos) {
          continue;
      }

      /*
      Extract the substring that starts from the last hip substring to the
      end of that string (until the open bracket from the original string).
      Remove all spaces if they exist in the parsed string, e.g.,
      hipError_t hipDeviceSetLimit ( enum hipLimit_t limit, size_t value );.
      */
      std::string api_name{api_name_no_brackets.substr(api_name_no_brackets.rfind(hip_api_prefix))};
      api_name.erase(std::remove(api_name.begin(), api_name.end(), ' '), api_name.end());

      if (!api_group_names_tracker.empty()) {
        /*
        If the API is not present in the global list of HIP APIs, add it.
        */
        HipAPI hip_api{api_name, deprecated_flag, api_group_names_tracker.back()};
        if(std::find(hip_apis.begin(), hip_apis.end(), hip_api) == hip_apis.end())
        {
            hip_apis.push_back(hip_api);
        }
      } else {
        std::cout << "[SKIP_FROM_COV] Group not detected for \"" << api_name << "\" in file \""
            << hip_api_header_file << "\", line " << line_number << std::endl;
      }
    }
  }

  hip_header_file_handler.close();
  return hip_apis;
}

/*
Used to extract test .cc files from the passed tests root directory.
Goes through all subdirectories and searches for .cc and .hh files for
HIP API invocations.
Implements BFS algorithm to avoid recursion.
*/
std::vector<std::string> extractTestModuleFiles(std::string& tests_root_directory)
{
  std::vector<std::string> directory_queue;
  directory_queue.push_back(tests_root_directory);
  std::vector<std::string> test_module_files;

  while (!directory_queue.empty()) {
    std::string processed_entry = directory_queue.back();
    directory_queue.pop_back();
    for (const auto& entry: std::filesystem::directory_iterator(processed_entry)) {
      if (std::filesystem::is_directory(entry.path())) {
        directory_queue.push_back(entry.path());
      } else {
        if (entry.path().string().find(".cc") != std::string::npos || 
            entry.path().string().find(".hh") != std::string::npos) {
          test_module_files.push_back(entry.path());
        }
      }
    }
  }

  return test_module_files;
}

std::string findAbsolutePathOfFile(std::string file_path)
{
  return std::filesystem::canonical(std::filesystem::absolute(file_path));
}
