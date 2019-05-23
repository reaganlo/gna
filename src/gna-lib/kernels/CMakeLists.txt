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
cmake_minimum_required(VERSION 3.10)

# global variables
set(PROJECT_NAME kernels)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

# SIMD compiler flags
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CXX_AVX2_FLAG "-march=core-avx2")
    set(CXX_AVX_FLAG "-mavx")
    set(CXX_SSE4_FLAG "-msse4.2")
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(CXX_AVX2_FLAG "/arch:CORE-AVX2")
    set(CXX_AVX_FLAG "/arch:AVX")
    set(CXX_SSE4_FLAG "/arch:SSE4.2")
  endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSCV")
  set(CXX_AVX2_FLAG "/arch:AVX2")
  set(CXX_AVX_FLAG "/arch:AVX")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CXX_AVX2_FLAG "-mavx2")
  set(CXX_AVX_FLAG "-mavx")
  set(CXX_SSE4_FLAG "-msse4.2")
endif()

set(KERNEL_DIR ${SRC_DIR}/kernels)

# --- XNN KERNELS --- #
set(xnn_kernel_sources
  isbmm8.cpp
  isbmm16.cpp
  pwl.cpp
  XnnKernel.cpp
  KernelArguments.cpp)

set(xnn_kernel_headers
  ${COMMON_DIR}/GnaException.h
  ${COMMON_DIR}/Macros.h
  convnet.h
  igemv.h
  igemv8.h
  igemv16.h
  KernelMacros.h
  pwl.h)

set(xnn_kernel_api
  ${KERNEL_DIR}/XnnKernel.h
  ${KERNEL_DIR}/KernelArguments.h)

set(xnn_generic_sources
  convnet_generic.cpp
  igemm16_generic.cpp
  igemm16_subset_generic.cpp
  igemm8_generic.cpp
  igemm8_subset_generic.cpp
  igemv16_generic.cpp
  igemv8_generic.cpp
  transpose16_generic.cpp)

set(xnn_generic_sat_sources
  convnet_generic-sat.cpp
  igemm16_generic-sat.cpp
  igemm16_subset_generic-sat.cpp
  igemm8_generic-sat.cpp
  igemm8_subset_generic-sat.cpp
  igemv16_generic-sat.cpp
  igemv8_generic-sat.cpp
  transpose16_generic.cpp)

set(xnn_sse4_sources
  convnet_sse4.cpp
  igemm16_sse4.cpp
  igemm16_subset_sse4.cpp
  igemm8_sse4.cpp
  igemm8_subset_sse4.cpp
  igemv16_sse4.cpp
  igemv8_sse4.cpp
  transpose16_sse4.cpp)

set(xnn_sse4_sat_sources
  convnet_sse4-sat.cpp
  igemm16_sse4-sat.cpp
  igemm16_subset_sse4-sat.cpp
  igemm8_sse4-sat.cpp
  igemm8_subset_sse4-sat.cpp
  igemv16_sse4-sat.cpp
  igemv8_sse4-sat.cpp
  transpose16_sse4.cpp)

set(xnn_avx1_sources
  convnet_avx1.cpp
  igemm16_avx1.cpp
  igemm16_subset_avx1.cpp
  igemm8_avx1.cpp
  igemm8_subset_avx1.cpp
  igemv16_avx1.cpp
  igemv8_avx1.cpp
  transpose16_avx1.cpp)

set(xnn_avx1_sat_sources
  convnet_avx1-sat.cpp
  igemm16_avx1-sat.cpp
  igemm16_subset_avx1-sat.cpp
  igemm8_avx1-sat.cpp
  igemm8_subset_avx1-sat.cpp
  igemv16_avx1-sat.cpp
  igemv8_avx1-sat.cpp
  transpose16_avx1.cpp)

set(xnn_avx2_sources
  convnet_avx2.cpp
  igemm16_avx2.cpp
  igemm16_subset_avx2.cpp
  igemm8_avx2.cpp
  igemm8_subset_avx2.cpp
  igemv16_avx2.cpp
  igemv8_avx2.cpp
  transpose16_avx2.cpp)

set(xnn_avx2_sat_sources
  convnet_avx2-sat.cpp
  igemm16_avx2-sat.cpp
  igemm16_subset_avx2-sat.cpp
  igemm8_avx2-sat.cpp
  igemm8_subset_avx2-sat.cpp
  igemv16_avx2-sat.cpp
  igemv8_avx2-sat.cpp
  transpose16_avx2.cpp)

add_library(xnn_kernel_generic STATIC)
target_include_directories(xnn_kernel_generic
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_generic
  PRIVATE ${xnn_kernel_sources} ${xnn_generic_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_generic_sat STATIC)
target_include_directories(xnn_kernel_generic_sat
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_generic_sat
  PRIVATE ${xnn_kernel_sources} ${xnn_generic_sat_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_sse4 STATIC)
target_include_directories(xnn_kernel_sse4
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_sse4
  PRIVATE ${xnn_kernel_sources} ${xnn_sse4_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_sse4_sat STATIC)
target_include_directories(xnn_kernel_sse4_sat
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_sse4_sat
  PRIVATE ${xnn_kernel_sources} ${xnn_sse4_sat_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_avx1 STATIC)
target_include_directories(xnn_kernel_avx1
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_avx1
  PRIVATE ${xnn_kernel_sources} ${xnn_avx1_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_avx1_sat STATIC)
target_include_directories(xnn_kernel_avx1_sat
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_avx1_sat
  PRIVATE ${xnn_kernel_sources} ${xnn_avx1_sat_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_avx2 STATIC)
target_include_directories(xnn_kernel_avx2
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_avx2
  PRIVATE ${xnn_kernel_sources} ${xnn_avx2_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

add_library(xnn_kernel_avx2_sat STATIC)
target_include_directories(xnn_kernel_avx2_sat
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(xnn_kernel_avx2_sat
  PRIVATE ${xnn_kernel_sources} ${xnn_avx2_sat_sources} ${xnn_kernel_headers}
  PUBLIC ${xnn_kernel_api})

target_compile_definitions(xnn_kernel_generic
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTGEN)

target_compile_definitions(xnn_kernel_generic_sat
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTGEN_SAT)

target_compile_definitions(xnn_kernel_sse4
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTSSE4)

target_compile_definitions(xnn_kernel_sse4_sat
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTSSE4_SAT)

target_compile_definitions(xnn_kernel_avx1
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTAVX1)

target_compile_definitions(xnn_kernel_avx1_sat
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTAVX1_SAT)

target_compile_definitions(xnn_kernel_avx2
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTAVX2)

target_compile_definitions(xnn_kernel_avx2_sat
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTAVX2_SAT)

target_compile_options(xnn_kernel_generic
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>)

target_compile_options(xnn_kernel_generic_sat
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>)

target_compile_options(xnn_kernel_sse4
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_SSE4_FLAG})

target_compile_options(xnn_kernel_sse4_sat
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_SSE4_FLAG})

target_compile_options(xnn_kernel_avx1
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_AVX_FLAG})

target_compile_options(xnn_kernel_avx1_sat
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_AVX_FLAG})

target_compile_options(xnn_kernel_avx2
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_AVX2_FLAG})

target_compile_options(xnn_kernel_avx2_sat
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_AVX2_FLAG})

set(xnn_kernel_libraries
  xnn_kernel_generic
  xnn_kernel_generic_sat
  xnn_kernel_sse4
  xnn_kernel_sse4_sat
  xnn_kernel_avx1
  xnn_kernel_avx1_sat
  xnn_kernel_avx2
  xnn_kernel_avx2_sat)

set_property(TARGET ${xnn_kernel_libraries} PROPERTY FOLDER library/kernels/xnn)

# ___ GMM KERNELS ___ #
set(gmm_kernel_sources
  gmm.cpp
  kernel-gmm.cpp
  KernelArguments.cpp)

set(gmm_kernel_headers
  ${COMMON_DIR}/Macros.h
  ${COMMON_DIR}/common.h
  ${COMMON_DIR}/GnaException.h
  KernelMacros.h
  kernel-gmm.h)

set(gmm_kernel_api
  ${KERNEL_DIR}/gmm.h
  ${KERNEL_DIR}/KernelArguments.h)

add_library(gmm_kernel_generic STATIC)
target_include_directories(gmm_kernel_generic
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(gmm_kernel_generic
  PRIVATE ${gmm_kernel_sources} ${gmm_kernel_headers}
  PUBLIC ${gmm_kernel_api})

add_library(gmm_kernel_sse4    STATIC)
target_include_directories(gmm_kernel_sse4
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(gmm_kernel_sse4
  PRIVATE ${gmm_kernel_sources} ${gmm_kernel_headers}
  PUBLIC ${gmm_kernel_api})

add_library(gmm_kernel_avx1    STATIC)
target_include_directories(gmm_kernel_avx1
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(gmm_kernel_avx1
  PRIVATE ${gmm_kernel_sources} ${gmm_kernel_headers}
  PUBLIC ${gmm_kernel_api})

add_library(gmm_kernel_avx2    STATIC)
target_include_directories(gmm_kernel_avx2
  PRIVATE ${COMMON_DIR} ${API_DIR} ${KERNEL_DIR} ${API_IMPL_DIR})
target_sources(gmm_kernel_avx2
  PRIVATE ${gmm_kernel_sources} ${gmm_kernel_headers}
  PUBLIC ${gmm_kernel_api})

target_compile_definitions(gmm_kernel_generic
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTGEN)

target_compile_options(gmm_kernel_generic
  PRIVATE
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${GNA_COMPILE_FLAGS})

target_compile_definitions(gmm_kernel_sse4
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTSSE4)

target_compile_options(gmm_kernel_sse4
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_SSE4_FLAG})

target_compile_definitions(gmm_kernel_avx1
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTAVX1)

target_compile_options(gmm_kernel_avx1
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_AVX_FLAG})

target_compile_definitions(gmm_kernel_avx2
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  OPTAVX2)

target_compile_options(gmm_kernel_avx2
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
  ${CXX_AVX2_FLAG})

set(gmm_kernel_libraries
  gmm_kernel_generic
  gmm_kernel_sse4
  gmm_kernel_avx1
  gmm_kernel_avx2)

set_property(TARGET ${gmm_kernel_libraries} PROPERTY FOLDER library/kernels/gmm)

set(kernel_libraries
  ${gmm_kernel_libraries} ${xnn_kernel_libraries}
  PARENT_SCOPE)

set(kernel_api_headers
  ${KERNEL_DIR}/gmm.h
  ${KERNEL_DIR}/XnnKernel.h
  PARENT_SCOPE)

set_target_properties(xnn_kernel_generic xnn_kernel_generic_sat
  xnn_kernel_sse4 xnn_kernel_sse4_sat xnn_kernel_avx1 xnn_kernel_avx1_sat
  xnn_kernel_avx2 xnn_kernel_avx2_sat
  gmm_kernel_generic gmm_kernel_sse4 gmm_kernel_avx1 gmm_kernel_avx2
  PROPERTIES
  POSITION_INDEPENDENT_CODE True
  PREFIX "")
