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
set(PROJECT_NAME gna-lib)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# project name
project(${PROJECT_NAME})

include(CTest)
# main binary path
set(GNA_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

option(GNA_RUN_CPPCHECK "Run cppcheck analysis" OFF)
if(GNA_RUN_CPPCHECK)
  set(CMAKE_CXX_CPPCHECK cppcheck)
  set(CMAKE_C_CPPCHECK cppcheck)
endif()

option(GNA_RUN_IWYU "Run iwyu analysis" OFF)
if(GNA_RUN_IWYU)
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE iwyu)
  set(CMAKE_C_INCLUDE_WHAT_YOU_USE iwyu)
endif()

option(GNA_RUN_CLANG_TIDY "Run clang-tidy analysis" OFF)
if(GNA_RUN_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*,portability-*,clang-analyzer-*)
endif()

if(DEFINED ENV{GNA_LIBRARY_VERSION})
  set(GNA_LIBRARY_VER -DGNA_LIBRARY_VERSION_STRING=\"$ENV{GNA_LIBRARY_VERSION}\")
elseif(NOT DEFINED GNA_LIBRARY_VER)
  set(GNA_LIBRARY_VER -DGNA_LIBRARY_VERSION_STRING=\"2.X.0.0\")
endif()

# source path variables
set(APP_DIR ${CMAKE_SOURCE_DIR}/src)
set(COMMON_TOOLS_DIR ${CMAKE_SOURCE_DIR}/src/tools/common)
set(COMMON_DIR ${APP_DIR}/common)
set(API_DIR ${APP_DIR}/gna-api)
set(SRC_DIR ${APP_DIR}/gna-lib)
set(API_IMPL_DIR ${APP_DIR}/gna-lib/gna-api)
set(KERNEL_DIR ${SRC_DIR}/kernels)
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)

# linking with libpthreads
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR
  ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  find_package(Threads)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(OS_PREFIX WIN)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(OS_PREFIX AOS)
else()
  set(OS_PREFIX LNX)
endif()

# setting default configuration
include(${CMAKE_SOURCE_DIR}/build/cmake/set_build_type.cmake)

# Visual Studio
if(MSVC)
  string(TOLOWER ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID} CMAKE_ARCHITECTURE)
elseif(NOT CMAKE_ARCHITECTURE)
  set(CMAKE_ARCHITECTURE "x64")
endif()

include(${CMAKE_SOURCE_DIR}/build/cmake/gna_compile_flags.cmake)

include(${CMAKE_SOURCE_DIR}/build/cmake/gna_cmake_tools.cmake)

file(COPY readme.txt
    DESTINATION ${GNA_BINARY_DIR}/)

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
  message(STATUS "Found ctags: ${CTAGS_PATH}")

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    execute_process(
      COMMAND
      bash -c "echo | gcc -Wp,-v -x c++ - -fsyntax-only 2>&1 | grep ^./usr | xargs"
      OUTPUT_VARIABLE CXX_STD_INCLUDE_DIRS)
  else()
    set(CXX_STD_INCLUDE_DIRS "")
  endif()
  add_custom_target(tags
    COMMAND
    bash -c "ctags -R --c++-kinds=+p --fields=+iaS --extra=+q ${CMAKE_SOURCE_DIR}/src ${CXX_STD_INCLUDE_DIRS}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

find_program(CSCOPE_PATH cscope)
if(CSCOPE_PATH)
  message(STATUS "Found cscope: ${CSCOPE_PATH}")

  add_custom_target(cscope
    COMMAND find -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" > cscope.files
    COMMAND cscope -Rqb -i cscope.files
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# HW C-package module
option(GNA_BUILD_WITH_3_0_HW_MODULE_ENABLED "" ON)

set(HW_MODULE_3_0_PATH)
set(GNA_HW_LIB_3_0_INCLUDE_PATH)
set(GNA_HW_LIB_ENABLED )

if(${GNA_BUILD_WITH_3_0_HW_MODULE_ENABLED})
  set(HW_MODULE_3_0_PATH ${BUILD_DIR}/hw_module_3_0)
  set(GNA_HW_LIB_3_0_INCLUDE_PATH "${HW_MODULE_3_0_PATH}/GNA-3d0_ArchCPkg_Lib/Include/")
  set(GNA_HW_LIB_ENABLED "-DGNA_HW_LIB_ENABLED=1")
  message("Enabling GNA 3.0 HW Module for GNA library")
  if(NOT EXISTS "${HW_MODULE_3_0_PATH}/CMakeLists.txt")

    message(FATAL_ERROR "HW 3.0 submodule was not initialized properly.\n"
        "File <${HW_MODULE_3_0_PATH}/CMakeLists.txt> not found.\n"
        "Try using 'git submodule update --init --recursive' to load all submodules.")
  endif()
  add_subdirectory(${HW_MODULE_3_0_PATH})
  set_gna_compile_definitions(gna_hw)
  gna_add_shared_library_rc_properties(gna_hw "Hardware support module")
else()
  message("HW Module 3.0 for GNA library will NOT be used.")
endif()

option(GNA_BUILD_WITH_3_5_HW_MODULE_ENABLED "" ON)
set(HW_MODULE_3_5_PATH)
set(GNA_HW_LIB_3_5_INCLUDE_PATH)
set(GNA_HW_LIB_3_5_ENABLED )


if(${GNA_BUILD_WITH_3_5_HW_MODULE_ENABLED})
  set(HW_MODULE_3_5_PATH ${BUILD_DIR}/hw_module_3_5)
  set(GNA_HW_LIB_3_5_INCLUDE_PATH "${HW_MODULE_3_5_PATH}/GNA-3d0_ArchCPkg/GNA-3d0_ArchCPkg_Lib/Include/")
  set(GNA_HW_LIB_ENABLED "-DGNA_HW_LIB_ENABLED=1")
  message("Enabling GNA 3.5 HW Module for GNA library")
  if(NOT EXISTS "${HW_MODULE_3_5_PATH}/CMakeLists.txt")

    message(FATAL_ERROR "HW 3.5 submodule was not initialized properly.\n"
        "File <${HW_MODULE_3_5_PATH}/CMakeLists.txt> not found.\n"
        "Try using 'git submodule update --init --recursive' to load all submodules.")
  endif()
  add_subdirectory(${HW_MODULE_3_5_PATH})
  set_gna_compile_definitions(gna-hw-3-5)
  gna_add_shared_library_rc_properties(gna-hw-3-5 "Hardware support module")
else()
  message("HW Module 3.5 for GNA library will NOT be used.")
endif()

set(GNA_HW_LIB_INCLUDE_PATH ${GNA_HW_LIB_3_0_INCLUDE_PATH} ${GNA_HW_LIB_3_5_INCLUDE_PATH})

add_subdirectory(src/gna-lib/kernels)
add_subdirectory(src/gna-lib)
add_subdirectory(src/tools)

# Set up gtest unit testing engine
add_subdirectory(src/gtest)
