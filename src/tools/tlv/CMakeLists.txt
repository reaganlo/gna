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

project(Tlv)

if (NOT DEFINED GNA_TOOLS_DEBUG_OUT_DIR)
  set(GNA_TLV_DEBUG_OUT_DIR ${PROJECT_BINARY_DIR}/tlv/${CMAKE_ARCHITECTURE})
  set(GNA_TLV_RELEASE_OUT_DIR ${PROJECT_BINARY_DIR}/tlv/${CMAKE_ARCHITECTURE})
  set(GNA_TLV_API_DEBUG_INCL_DIR ${PROJECT_BINARY_DIR}/tlv/include)
  set(GNA_TLV_API_RELEASE_INCL_DIR ${PROJECT_BINARY_DIR}/tlv/include)
else()
  set(GNA_TLV_DEBUG_OUT_DIR ${GNA_TOOLS_DEBUG_OUT_DIR}/tlv/${CMAKE_ARCHITECTURE})
  set(GNA_TLV_RELEASE_OUT_DIR ${GNA_TOOLS_RELEASE_OUT_DIR}/tlv/${CMAKE_ARCHITECTURE})
  set(GNA_TLV_API_DEBUG_INCL_DIR ${GNA_TOOLS_DEBUG_OUT_DIR}/tlv/include)
  set(GNA_TLV_API_RELEASE_INCL_DIR ${GNA_TOOLS_RELEASE_OUT_DIR}/tlv/include)
endif()

set(tlv_src
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvRead.c
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvWrite.c
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvErrors.c)

set(tlv_api
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvApi.h)

set(tlv_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvRead.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvWrite.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvErrors.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TlvCommon.h)

add_library(tlv STATIC
    ${tlv_src}
    ${tlv_headers}
    ${tlv_api})

set_target_properties(tlv
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_TLV_DEBUG_OUT_DIR}
  LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_TLV_RELEASE_OUT_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_TLV_DEBUG_OUT_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_TLV_RELEASE_OUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_TLV_DEBUG_OUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_TLV_RELEASE_OUT_DIR}
  EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
  C_STANDARD 11
  OUTPUT_NAME tlv)

target_compile_definitions(tlv
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>)

target_compile_options(tlv
  PRIVATE
  ${GNA_CC_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_CC_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_CC_COMPILE_FLAGS_RELEASE}>
  ${EXTRA_EXE_COMPILE_OPTIONS})

if(NOT ${GNA_TLV_API_DEBUG_INCL_DIR})
  file(MAKE_DIRECTORY ${GNA_TLV_API_DEBUG_INCL_DIR})
endif()
if (NOT ${GNA_TLV_API_RELEASE_INCL_DIR})
  file(MAKE_DIRECTORY ${GNA_TLV_API_RELEASE_INCL_DIR})
endif()

add_custom_command(TARGET tlv POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${tlv_api}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_TLV_API_DEBUG_INCL_DIR}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_TLV_API_RELEASE_INCL_DIR}>
  $<$<CONFIG:KLOCWORK>:${GNA_TLV_API_RELEASE_INCL_DIR}>
  $<$<CONFIG:Debug>:${GNA_TLV_API_DEBUG_INCL_DIR}>
  $<$<CONFIG:Release>:${GNA_TLV_API_RELEASE_INCL_DIR}>)
