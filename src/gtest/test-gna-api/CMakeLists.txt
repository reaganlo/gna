#INTEL CONFIDENTIAL
#Copyright 2019 Intel Corporation.

#The source code contained or described herein and all documents related
#to the source code ("Material") are owned by Intel Corporation or its suppliers
#or licensors. Title to the Material remains with Intel Corporation or its suppliers
#and licensors. The Material may contain trade secrets and proprietary
#and confidential information of Intel Corporation and its suppliers and licensors,
#and is protected by worldwide copyright and trade secret laws and treaty provisions.
#No part of the Material may be used, copied, reproduced, modified, published,
#uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
#prior express written permission.

#No license under any patent, copyright, trade secret or other intellectual
#property right is granted to or conferred upon you by disclosure or delivery
#of the Materials, either expressly, by implication, inducement, estoppel
#or otherwise. Any license under such intellectual property rights must
#be express and approved by Intel in writing.

#Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
#or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
#in any way.

# minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# global variables
unset(PREFIX)
set(PROJECT_NAME TEST-API-VERBOSE)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

add_executable(test-gna-api
  test-gna-api.cpp)
    
target_include_directories(test-gna-api
  PRIVATE ${SRC_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR})
  
target_compile_definitions(test-gna-api PRIVATE -DOPTGEN)

target_link_libraries(test-gna-api gtest gna-api)

set_target_properties(test-gna-api
  PROPERTIES
  EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
  OUTPUT_NAME test-gna-api)
set_property(TARGET test-gna-api PROPERTY FOLDER tests)

add_custom_command(TARGET test-gna-api POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  $<TARGET_FILE:gna-api>
  $<$<CONFIG:LNX_RELEASE>:$<TARGET_FILE:gna-api>.dbg>
  $<TARGET_FILE_DIR:test-gna-api>)

add_target_to_postbuild_test_run(test-gna-api)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  add_executable(test-gna-api-verbose
      test-gna-api-verbose.cpp
  )

  target_link_libraries(test-gna-api-verbose gtest gna-api-verbose)
  set_target_properties(test-gna-api-verbose
    PROPERTIES
    FOLDER tests
    EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
    OUTPUT_NAME test-gna-api-verbose)

  add_target_to_postbuild_test_run(test-gna-api-verbose)
endif()
