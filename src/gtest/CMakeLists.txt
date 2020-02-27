#INTEL CONFIDENTIAL
#Copyright 2020 Intel Corporation.

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

cmake_minimum_required(VERSION 3.10)

option(GNA_BUILD_WITH_GTEST_ENABLED "" ON)
if(${GNA_BUILD_WITH_GTEST_ENABLED})
  message("Enabling gtest for GNA library")

  if(NOT EXISTS "${BUILD_DIR}/googletest/CMakeLists.txt")
    find_package(Git)
    if(NOT GIT_FOUND)
      message(FATAL_ERROR "ERROR: Git package have not been found")
    endif()
    message("Executing '${GIT_EXECUTABLE} submodule update --init --recursive' command")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GNA_GIT_SUBMODULE_UPDATE_OUTPUT_VARIABLE
      ERROR_VARIABLE GNA_GIT_SUBMODULE_UPDATE_ERROR_VARIABLE
      RESULT_VARIABLE GNA_GIT_SUBMODULE_UPDATE_RESULT_VARIABLE)
    if(GNA_GIT_SUBMODULE_UPDATE_OUTPUT_VARIABLE)
      message("git's stdout message: [ ${GNA_GIT_SUBMODULE_UPDATE_OUTPUT_VARIABLE} ]")
    endif()
    if(GNA_GIT_SUBMODULE_UPDATE_ERROR_VARIABLE)
      message("git's stderr message: [ ${GNA_GIT_SUBMODULE_UPDATE_ERROR_VARIABLE} ]")
    endif()
    if(NOT GNA_GIT_SUBMODULE_UPDATE_RESULT_VARIABLE EQUAL "0")
      message("WARNING: git's result: [ ${GNA_GIT_SUBMODULE_UPDATE_RESULT_VARIABLE} ]")
    endif()
    if(NOT EXISTS "${BUILD_DIR}/googletest/CMakeLists.txt")
      message("ERROR: File <${BUILD_DIR}/googletest/CMakeLists.txt> not found")
      message(FATAL_ERROR "ERROR: googletest submodule was not initialized properly")
    endif()
  endif()

  add_subdirectory("${BUILD_DIR}/googletest" "out_binary_googletest")

  # Enable 'gtest_add_tests' command
  include(GoogleTest)

  add_custom_target(gna_all_tests_to_run_on_posbuild_via_ctest ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:gna-api>
      $<$<CONFIG:LNX_RELEASE>:$<TARGET_FILE:gna-api>.dbg>
      $<TARGET_FILE_DIR:test-gna-api>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:gna-api>
      $<$<CONFIG:LNX_RELEASE>:$<TARGET_FILE:gna-api>.dbg>
      $<TARGET_FILE_DIR:test-model-error>)
  set_property(TARGET gna_all_tests_to_run_on_posbuild_via_ctest PROPERTY FOLDER tests/common)
  if(NOT "${GNA_ALL_TESTS_REGISTERED_TARGETS}" STREQUAL "")
    add_dependencies(gna_all_tests_to_run_on_posbuild_via_ctest ${GNA_ALL_TESTS_REGISTERED_TARGETS})
  endif()

  macro(add_target_to_postbuild_test_run target_test)
    gtest_add_tests(
      TARGET ${target_test}
      TEST_PREFIX GNA_POST_BUILD_)
    add_dependencies(gna_all_tests_to_run_on_posbuild_via_ctest ${target_test})
  endmacro(add_target_to_postbuild_test_run)

  option(BUILD_GMOCK "" OFF)

  add_subdirectory(test-gna-api)
  add_subdirectory(test-layer)
  add_subdirectory(test-model-error)
  add_subdirectory(test-xnn)
  add_subdirectory(test-class)
  if (${GNA_BUILD_WITH_TLV_ENABLED})
    add_subdirectory(test-tlv)
  endif()

  set_property(TARGET gtest PROPERTY FOLDER tests/common)
  set_property(TARGET gtest_main PROPERTY FOLDER tests/common)
  if (${BUILD_GMOCK} STREQUAL "ON")
    set_property(TARGET gmock PROPERTY FOLDER tests/common)
    set_property(TARGET gmock_main PROPERTY FOLDER tests/common)
  endif()

  option(GNA_RUN_TESTS_ON_BUILD "" ON)
  if(GNA_RUN_TESTS_ON_BUILD)
    if(CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME)
      add_custom_command(
        TARGET gna_all_tests_to_run_on_posbuild_via_ctest
        POST_BUILD
        COMMENT "Run all registered tests via ctest"
        COMMAND ctest ARGS -R GNA_POST_BUILD_ -C "$<CONFIG>" --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      endif()
  endif()
endif()
