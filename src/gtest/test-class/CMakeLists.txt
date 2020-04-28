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

# global variables
unset(PREFIX)
set(PROJECT_NAME test-class)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(EXTRA_LIBS ${CMAKE_DL_LIBS})
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(API_EXTRA_DEFS "-DINTEL_GNA_DLLEXPORT=1" ${GNA_HW_LIB_ENABLED})
endif()

file(GLOB lib_sources
  "${SRC_DIR}/*HwModuleInterface*.hpp"
  "${SRC_DIR}/*HwModuleInterface*.cpp"
  "${SRC_DIR}/*Logger*.cpp"
  "${API_IMPL_DIR}/gna2-common-impl.*"
  "${SRC_DIR}/Expect.*"
  "${SRC_DIR}/GnaException.*"
  "${SRC_DIR}/DataMode.*"
  )

file(GLOB sources
  "*.hpp"
  "*.cpp")

add_executable(${PROJECT_NAME}
  ${sources}
  ${lib_sources}
   )

target_compile_definitions(${PROJECT_NAME}
  PRIVATE ${EXTRA_DEFS} ${API_EXTRA_DEFS} PROFILE PROFILE_DETAILED)

target_include_directories(${PROJECT_NAME}
  PRIVATE ${COMMON_DIR} ${SRC_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR} ${GNA_HW_LIB_INCLUDE_PATH})
  
target_link_libraries(${PROJECT_NAME}
  gtest
  ${EXTRA_LIBS})

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
  OUTPUT_NAME ${PROJECT_NAME})

set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER tests/gna-lib)

add_target_to_postbuild_test_run(${PROJECT_NAME})
