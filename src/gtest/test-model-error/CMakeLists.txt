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

# minimum version of CMake
cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME TEST-MODEL-ERROR)
set(TARGET_NAME test-model-error)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

add_executable(${TARGET_NAME}
  test-main.cpp
  test-model-error.cpp
  test-model-error.hpp)

target_include_directories(${TARGET_NAME}
  PRIVATE ${SRC_DIR} ${COMMON_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR} ${GNA_HW_LIB_INCLUDE_PATH} ${COMMON_TOOLS_DIR})

target_compile_definitions(${TARGET_NAME} PRIVATE -DOPTGEN)

target_link_libraries(${TARGET_NAME} gtest gna-api)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
  OUTPUT_NAME ${TARGET_NAME})
set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER tests/gna-lib)

add_target_to_postbuild_test_run(${TARGET_NAME})

