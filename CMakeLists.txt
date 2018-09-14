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
set(PROJECT_NAME gna-lib)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# project name
project(${PROJECT_NAME})

# main binary path
set(GNA_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

# source path variables
set(APP_DIR ${CMAKE_SOURCE_DIR}/src)
set(COMMON_DIR ${APP_DIR}/common)
set(SRC_DIR ${APP_DIR}/gna-lib)
set(SAMPLES_DIR ${CMAKE_SOURCE_DIR}/samples)
set(KERNEL_DIR ${SRC_DIR}/kernels)

# output path variables
set(GNA_LIB_OUT_DIR ${GNA_BINARY_DIR}/gna-lib/$<CONFIG>)

# linking with libpthreads
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  find_package(Threads)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(OS_PREFIX WIN)
else()
  set(OS_PREFIX LNX)
endif()

# setting default configuration
include(${CMAKE_SOURCE_DIR}/build/cmake/set_build_type.cmake)

# postfix for debug libraries
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(CMAKE_${OS_PREFIX}-DEBUG_POSTFIX d)
endif()

# Visual Studio
if(MSVC)
  string(TOLOWER ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID} CMAKE_ARCHITECTURE)
elseif(NOT CMAKE_ARCHITECTURE)
  set(CMAKE_ARCHITECTURE "x64")
endif()

# configuration flags
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(CXX_FLAGS "-g -Wall")
  set(CXX_FLAGS_DEBUG "-O0 -DDEBUG=1")
  set(CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} -fbuiltin -inline-forceinline")
  endif()
  set(CXX_LINKER_FLAGS_RELEASE "-fdata-sections -ffunction-sections -Wl,--gc-sections")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(CXX_FLAGS "/EHa /DWIN32 /D_WINDOWS /Zi /Wall /sdl")
  set(CXX_FLAGS_DEBUG "/MDd /Od /DDEBUG=1 /RTC1")
  set(CXX_FLAGS_RELEASE "/MD /O3 /Oi /DNDEBUG /Gy")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} /Qinline-forceinline")
  endif()
  set(CXX_LINKER_FLAGS "/DEBUG")
  set(CXX_LINKER_FLAGS_DEBUG "/INCREMENTAL")
  set(CXX_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO /NOLOGO /OPT:REF /OPT:ICF")
endif()

# interprocedural optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION_${OS_PREFIX}-RELEASE TRUE)
else()
  message(WARNING "IPO is not supported: ${output}. Setting flags manually")

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} /Qipo")
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
      set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} /GL")
    endif()
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} -ipo")
    else()
      set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} -flto -fno-fat-lto-objects")
    endif()
  endif()
endif()

# set 32-bit compilation flags
if(CMAKE_ARCHITECTURE STREQUAL x86)
  include(${CMAKE_SOURCE_DIR}/build/cmake/set_x86_flags.cmake)
endif()

set(CMAKE_CXX_FLAGS                                ${CXX_FLAGS})
set(CMAKE_CXX_FLAGS_${OS_PREFIX}-DEBUG             ${CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_${OS_PREFIX}-RELEASE           ${CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_KLOCWORK                      ${CXX_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS                      "${CMAKE_SHARED_LINKER_FLAGS} ${CXX_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}-DEBUG   ${CXX_LINKER_FLAGS_DEBUG})
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}-RELEASE ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS_KLOCWORK            ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_EXE_LINKER_FLAGS                         "${CMAKE_EXE_LINKER_FLAGS} ${CXX_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}-DEBUG      ${CXX_LINKER_FLAGS_DEBUG})
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}-RELEASE    ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_EXE_LINKER_FLAGS_KLOCWORK               ${CXX_LINKER_FLAGS_RELEASE})

add_subdirectory(src/gna-lib/kernels)
add_subdirectory(src/gna-lib)
add_subdirectory(src/tools)
