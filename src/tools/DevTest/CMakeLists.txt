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
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(devtest_sources
  ChainModel.cpp
  DeviceController.cpp
  DevTest.cpp
  IModelSetup.cpp
  ModelSetup.cpp
  ModelSetupFactory.cpp
  ModelUtilities.cpp
  Script.cpp
  SetupConvolutionModel.cpp
  SetupConvolutionModel2D.cpp
  SetupCopyModel.cpp
  SetupDiagonalModel.cpp
  SetupDnnModel_1.cpp
  SetupDnnModel_Multibuffer.cpp
  SetupGmmModel.cpp
  SetupMixModel.cpp
  SetupMultibiasModel_1.cpp
  SetupPoolingModel.cpp
  SetupRecurrentModel.cpp
  SetupSplitModel.cpp
  SetupTransposeModel.cpp)

set(devtest_headers
  ${COMMON_DIR}/Macros.h
  ChainModel.h
  DeviceController.h
  IModelSetup.h
  ModelSetup.h
  ModelSetupFactory.h
  ModelUtilities.h
  Script.h
  SetupConvolutionModel.h
  SetupConvolutionModel2D.h
  SetupCopyModel.h
  SetupDiagonalModel.h
  SetupDnnModel_1.h
  SetupDnnModel_Multibuffer.h
  SetupGmmModel.h
  SetupMixModel.h
  SetupMultibiasModel_1.h
  SetupPoolingModel.h
  SetupRecurrentModel.h
  SetupSplitModel.h
  SetupTransposeModel.h)

add_executable(DevTest
  ${devtest_sources}
  ${devtest_headers})

target_include_directories(DevTest PRIVATE ${API_DIR} ${COMMON_TOOLS_DIR})
target_link_libraries(DevTest gna-api)

set_gna_target_properties(DevTest)
set_gna_compile_definitions(DevTest)
set_gna_compile_options(DevTest)

copy_gna_api(DevTest NEW_TARGET_NAME)
copy_pdb_windows(DevTest)
strip_symbols(DevTest)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows" OR
   ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  enable_testing()
  add_test(NAME GNA_POST_BUILD_DevTest COMMAND DevTest)
  list(APPEND GNA_ALL_TESTS_REGISTERED_TARGETS ${NEW_TARGET_NAME})
  set(GNA_ALL_TESTS_REGISTERED_TARGETS "${GNA_ALL_TESTS_REGISTERED_TARGETS}" PARENT_SCOPE)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    target_link_libraries(DevTest ${ANDROID_CPP_LIB_SHARED_PATH})
endif()
