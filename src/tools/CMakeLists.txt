#INTEL CONFIDENTIAL
#Copyright 2018 Intel Corporation.

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
cmake_minimum_required(VERSION 3.9)

# global variables
unset(PREFIX)
set(PROJECT_NAME tools)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

set(GNA_TOOLS_OUT_DIR ${GNA_BINARY_DIR}/tools/$<CONFIG>)
set(GNA_DEVTEST_OUT_DIR ${GNA_TOOLS_OUT_DIR}/DevTest/${CMAKE_ARCHITECTURE})

set(devtest_sources
  DevTest/ChainModel.cpp
  DevTest/DeviceController.cpp
  DevTest/DevTest.cpp
  DevTest/ModelSetupFactory.cpp
  DevTest/ModelUtilities.cpp
  DevTest/Script.cpp
  DevTest/SetupConvolutionModel.cpp
  DevTest/SetupCopyModel.cpp
  DevTest/SetupDiagonalModel.cpp
  DevTest/SetupDnnModel_1.cpp
  DevTest/SetupGmmModel.cpp
  DevTest/SetupMixModel.cpp
  DevTest/SetupMultibiasModel_1.cpp
  DevTest/SetupPoolingModel.cpp
  DevTest/SetupRecurrentModel.cpp
  DevTest/SetupSplitModel.cpp
  DevTest/SetupTransposeModel.cpp)

set(devtest_headers
  DevTest/ChainModel.h
  DevTest/DeviceController.h
  DevTest/IModelSetup.h
  DevTest/ModelSetupFactory.h
  DevTest/ModelUtilities.h
  DevTest/Script.h
  DevTest/SetupConvolutionModel.h
  DevTest/SetupCopyModel.h
  DevTest/SetupDiagonalModel.h
  DevTest/SetupDnnModel_1.h
  DevTest/SetupGmmModel.h
  DevTest/SetupMixModel.h
  DevTest/SetupMultibiasModel_1.h
  DevTest/SetupPoolingModel.h
  DevTest/SetupRecurrentModel.h
  DevTest/SetupSplitModel.h
  DevTest/SetupTransposeModel.h)

add_executable(DevTest
  ${devtest_sources}
  ${devtest_headers})

target_include_directories(DevTest PRIVATE ${COMMON_DIR})

set_target_properties(DevTest
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}-DEBUG ${GNA_DEVTEST_OUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}-RELEASE ${GNA_DEVTEST_OUT_DIR}
    EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
    OUTPUT_NAME DevTest)

target_link_libraries(DevTest gna-api)

add_custom_command(TARGET DevTest POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  $<TARGET_FILE:gna-api>
  $<TARGET_FILE_DIR:DevTest>)

# on Windows copy pdb file
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  add_custom_command(TARGET DevTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_PDB_FILE:gna-api>
    $<TARGET_FILE_DIR:DevTest>)
endif()
