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

set(GNA_DRIVER_UTIL_DEBUG_OUT_DIR
  ${GNA_TOOLS_DEBUG_OUT_DIR}/gna-driver-util/${CMAKE_ARCHITECTURE})
set(GNA_DRIVER_UTIL_RELEASE_OUT_DIR
  ${GNA_TOOLS_RELEASE_OUT_DIR}/gna-driver-util/${CMAKE_ARCHITECTURE})

set(gna_driver_util_sources
  GnaUtilConfig.cpp
  gna-driver-util.cpp)

set(gna_driver_util_headers
  Memory.h
  GnaUtilConfig.h
  DriverInterface.h)

set(gna_driver_util_sources
  ${gna_driver_util_sources}
  WindowsDriverInterface.cpp
  LinuxDriverInterface.cpp)

set(gna_driver_util_headers
  ${gna_driver_util_headers}
  WindowsDriverInterface.h
  LinuxDriverInterface.h)

set(gna_driver_util_common_headers
  ${COMMON_DIR}/GnaDrvApi.h
  ${COMMON_DIR}/gna-api.h
  ${COMMON_DIR}/gna-api-status.h
  ${COMMON_DIR}/gna.h
  ${COMMON_DIR}/gna-h-wrapper.h)

add_executable(gna-driver-util
  ${gna_driver_util_common_headers}
  ${gna_driver_util_sources}
  ${gna_driver_util_headers})

target_include_directories(gna-driver-util
  PRIVATE ${COMMON_DIR} ${API_DIR})

set_gna_target_properties(gna-driver-util)
set_gna_compile_definitions(gna-driver-util)
set_gna_compile_options(gna-driver-util)

strip_symbols(gna-driver-util)
