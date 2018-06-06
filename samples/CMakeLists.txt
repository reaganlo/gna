# INTEL CONFIDENTIAL
# Copyright 2018 Intel Corporation.

# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its suppliers
# and licensors. The Material may contain trade secrets and proprietary
# and confidential information of Intel Corporation and its suppliers and licensors,
# and is protected by worldwide copyright and trade secret laws and treaty provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
# prior express written permission.

# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel
# or otherwise. Any license under such intellectual property rights must
# be express and approved by Intel in writing.

# Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
# or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
# in any way.

# minimum version of CMake
cmake_minimum_required(VERSION 3.9)

# global variables
set(PROJECT_NAME gna-samples)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# project name
project(${PROJECT_NAME})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  option(GNA_LOCAL_PACKAGE ON)
else()
  option(GNA_LOCAL_PACKAGE OFF)
endif()

# Visual Studio
if(MSVC)
  string(TOLOWER ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID} CMAKE_ARCHITECTURE)
elseif(NOT CMAKE_ARCHITECTURE)
  set(CMAKE_ARCHITECTURE "x64")
endif()

# output path variables
set(GNA_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(GNA_SAMPLE_OUT_DIR ${GNA_BINARY_DIR}/$<CONFIG>)

# configuration prefix
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(OS_PREFIX WIN)
else()
  set(OS_PREFIX LNX)
endif()

# configuration flags
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(CXX_FLAGS "-g -Wall")
  set(CXX_FLAGS_DEBUG "-O0 -DDEBUG=1")
  set(CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(CXX_FLAGS "/EHa /DWIN32 /D_WINDOWS /Zi /Wall /sdl")
  set(CXX_FLAGS_DEBUG "/MDd /Od /DDEBUG=1 /RTC1")
  set(CXX_FLAGS_RELEASE "/MD /O3 /Oi /DNDEBUG /Gy")
  set(CXX_LINKER_FLAGS "/DEBUG")
  set(CXX_LINKER_FLAGS_DEBUG "/INCREMENTAL")
  set(CXX_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO /NOLOGO /OPT:REF /OPT:ICF")
endif()

# set default configuration
include(${CMAKE_SOURCE_DIR}/../cmake/set_build_type.cmake)

# set 32-bit compilation flags
if(CMAKE_ARCHITECTURE STREQUAL x86)
  include(${CMAKE_SOURCE_DIR}/../cmake/set_x86_flags.cmake)
endif()

set(CMAKE_CXX_FLAGS                                ${CXX_FLAGS})
set(CMAKE_CXX_FLAGS_${OS_PREFIX}-DEBUG             ${CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_${OS_PREFIX}-RELEASE           ${CXX_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS                      "${CMAKE_SHARED_LINKER_FLAGS} ${CXX_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}-DEBUG   ${CXX_LINKER_FLAGS_DEBUG})
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}-RELEASE ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_EXE_LINKER_FLAGS                         "${CMAKE_EXE_LINKER_FLAGS} ${CXX_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}-DEBUG      ${CXX_LINKER_FLAGS_DEBUG})
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}-RELEASE    ${CXX_LINKER_FLAGS_RELEASE})

# create imported target or find GNA library in the system
if(GNA_LOCAL_PACKAGE)
  add_library(Gna::gna-api SHARED IMPORTED)
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_property(TARGET Gna::gna-api
      PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/gna-lib/${CMAKE_ARCHITECTURE}/gna.dll)
    set_property(TARGET Gna::gna-api
      PROPERTY IMPORTED_IMPLIB ${CMAKE_CURRENT_LIST_DIR}/gna-lib/${CMAKE_ARCHITECTURE}/gna.lib)
  else()
    set_property(TARGET Gna::gna-api
      PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/gna-lib/${CMAKE_ARCHITECTURE}/libgna.so.2)
  endif()
  set_property(TARGET Gna::gna-api PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/gna-lib/include)
  # possible in CMake 3.11
  # target_include_directories(Gna::gna-api INTERFACE ${CMAKE_SOURCE_DIR}/gna-lib/include)
else()
  find_package(Gna 2.0 REQUIRED)
endif()

# sample01
add_executable(sample01 ${CMAKE_SOURCE_DIR}/sample01/sample01.cpp)
target_include_directories(sample01 PRIVATE ${CMAKE_SOURCE_DIR}/sample01)
target_link_libraries(sample01 PRIVATE Gna::gna-api)
set_target_properties(sample01
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${GNA_BINARY_DIR}/sample01/$<CONFIG>/${CMAKE_ARCHITECTURE})

add_custom_command(TARGET sample01 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  $<TARGET_FILE:Gna::gna-api>
  $<TARGET_FILE_DIR:sample01>)

# sample_dumper
add_executable(sample_dumper ${CMAKE_SOURCE_DIR}/sample_dumper/sample_dumper.cpp)
target_include_directories(sample_dumper PRIVATE sample_dumper)
target_link_libraries(sample_dumper PRIVATE Gna::gna-api)
set_target_properties(sample_dumper
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${GNA_BINARY_DIR}/sample_dumper/$<CONFIG>/${CMAKE_ARCHITECTURE})

add_custom_command(TARGET sample_dumper POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  $<TARGET_FILE:Gna::gna-api>
  $<TARGET_FILE_DIR:sample_dumper>)
