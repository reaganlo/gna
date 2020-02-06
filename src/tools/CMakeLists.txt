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

set(GNA_TOOLS_OUT_DIR ${GNA_BINARY_DIR}/tools)
set(GNA_TOOLS_DEBUG_OUT_DIR ${GNA_BINARY_DIR}/tools/${OS_PREFIX}-DEBUG)
set(GNA_TOOLS_RELEASE_OUT_DIR ${GNA_BINARY_DIR}/tools/${OS_PREFIX}-RELEASE)

option(GNA_BUILD_WITH_DEVTEST_ENABLED "" ON)
option(GNA_BUILD_WITH_GNA_SELF_TEST_ENABLED "" ON)

include(${CMAKE_SOURCE_DIR}/build/cmake/gna_cmake_tools.cmake)

if(${GNA_BUILD_WITH_DEVTEST_ENABLED})
  add_subdirectory(DevTest)
endif()

if (${GNA_BUILD_WITH_GNA_SELF_TEST_ENABLED})
  add_subdirectory(gna-self-test)
endif()

option(GNA_BUILD_WITH_TLV_ENABLED "" ON)
if(${GNA_BUILD_WITH_TLV_ENABLED})
  add_subdirectory(tlv)
  set_gna_target_properties(tlv)
endif()

set(GNA_ALL_TESTS_REGISTERED_TARGETS "${GNA_ALL_TESTS_REGISTERED_TARGETS}" PARENT_SCOPE)
