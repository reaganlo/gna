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
set(CMAKE_C_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# project name
project(${PROJECT_NAME})
include(CTest)
# main binary path
set(GNA_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

# source path variables
set(APP_DIR ${CMAKE_SOURCE_DIR}/src)
set(COMMON_DIR ${APP_DIR}/common)
set(API_DIR ${APP_DIR}/gna-api)
set(SRC_DIR ${APP_DIR}/gna-lib)
set(SAMPLES_DIR ${CMAKE_SOURCE_DIR}/samples)
set(KERNEL_DIR ${SRC_DIR}/kernels)

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

# configuration flags
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR
  ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  if(${CMAKE_BUILD_TYPE} STREQUAL "${OS_PREFIX}_RELEASE")
    add_definitions("-D_FORTIFY_SOURCE=2")
  endif()

  set(CXX_FLAGS "-Wall -Werror")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    set(CXX_FLAGS "-Wall")
  endif()

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # clang double braces bug: https://bugs.llvm.org/show_bug.cgi?id=21629
    set(CXX_FLAGS "${CXX_FLAGS} -Wno-missing-braces")
  endif()
    set(CXX_FLAGS_DEBUG "-g -DDEBUG=1")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CXX_FLAGS_DEBUG "${CXX_FLAGS_DEBUG} -Og -DDEBUG=1")
  endif()

  set(CXX_FLAGS_RELEASE "-fvisibility=hidden -fvisibility-inlines-hidden -fstack-protector-all -O3 -DNDEBUG")

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} -fbuiltin")
    if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
      # Intel compiler hangs or crashes when using -inline-forceinline option for Android target
      # icpc: error #10014: problem during multi-file optimization compilation (code 4)

      # icpc: error #10106: Fatal error ....   (intel64/mcpcom), terminater by kill signal
      #set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} -inline-forceinline")
    endif()
  endif()
  set(CXX_LINKER_FLAGS "-z now")
  set(CXX_LINKER_FLAGS_RELEASE "-fdata-sections -ffunction-sections -Wl,--gc-sections")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(CXX_FLAGS "/EHa /DWIN32 /D_WINDOWS /Zi /WX /sdl")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CXX_FLAGS "${CXX_FLAGS} /W4")
  else()
    set(CXX_FLAGS "${CXX_FLAGS} /Wall")
  endif()
  set(CXX_FLAGS_DEBUG "/MDd /Od /DDEBUG=1 /RTC1")
  set(CXX_FLAGS_RELEASE "/MD /Oi /DNDEBUG /Gy")
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} /O3 /Qinline-forceinline")
  endif()
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} /O2")
  endif()
  set(CXX_LINKER_FLAGS "/DEBUG")
  set(CXX_LINKER_FLAGS_DEBUG "/INCREMENTAL")
  set(CXX_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO /NOLOGO /OPT:REF /OPT:ICF /PDBSTRIPPED:$(TargetDir)$(TargetName)Public.pdb")
endif()

# interprocedural optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(GLOBAL PROPERTY
    INTERPROCEDURAL_OPTIMIZATION_${OS_PREFIX}_RELEASE TRUE
    INTERPROCEDURAL_OPTIMIZATION_KLOCWORK TRUE)
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
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      # Intel compiler hangs or crashes when using -ipo option for Android target
      # icpc: error #10014: problem during multi-file optimization compilation (code 4)
      set(CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE} -flto -fno-fat-lto-objects")
    endif()
  endif()
endif()

# set 32-bit compilation flags
if(CMAKE_ARCHITECTURE STREQUAL x86)
  include(${CMAKE_SOURCE_DIR}/build/cmake/set_x86_flags.cmake)
endif()

# -fvisibility-inlines-hidden is a C++ only flag
string(REGEX REPLACE " -fvisibility-inlines-hidden" "" C_FLAGS_RELEASE ${CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS                                "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_${OS_PREFIX}_DEBUG             ${CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_${OS_PREFIX}_RELEASE           ${CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_KLOCWORK                       ${CXX_FLAGS_RELEASE})
set(CMAKE_C_FLAGS                                  "${CMAKE_C_FLAGS} ${CXX_FLAGS}")
set(CMAKE_C_FLAGS_${OS_PREFIX}_DEBUG               ${CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_${OS_PREFIX}_RELEASE             ${C_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_KLOCWORK                         ${C_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS                      "${CMAKE_SHARED_LINKER_FLAGS} ${CXX_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}_DEBUG   ${CXX_LINKER_FLAGS_DEBUG})
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}_RELEASE ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS_KLOCWORK             ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_EXE_LINKER_FLAGS                         "${CMAKE_EXE_LINKER_FLAGS} ${CXX_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}_DEBUG      ${CXX_LINKER_FLAGS_DEBUG})
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}_RELEASE    ${CXX_LINKER_FLAGS_RELEASE})
set(CMAKE_EXE_LINKER_FLAGS_KLOCWORK                ${CXX_LINKER_FLAGS_RELEASE})

# make samples package
file(MAKE_DIRECTORY ${GNA_BINARY_DIR}/samples)
file(MAKE_DIRECTORY ${GNA_BINARY_DIR}/samples/gna-lib)
file(MAKE_DIRECTORY ${GNA_BINARY_DIR}/samples/gna-lib/include)
file(MAKE_DIRECTORY ${GNA_BINARY_DIR}/samples/gna-lib/x64)
file(MAKE_DIRECTORY ${GNA_BINARY_DIR}/samples/gna-lib/x86)
file(COPY samples/CMakeLists.txt
          samples/gna-samples.sln
          samples/readme.txt
          samples/sample01
          samples/sample_dumper
    DESTINATION ${GNA_BINARY_DIR}/samples)
file(COPY readme.txt
    DESTINATION ${GNA_BINARY_DIR}/)

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
  message(STATUS "Found ctags: ${CTAGS_PATH}")

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    execute_process(
      COMMAND bash -c "echo | gcc -Wp,-v -x c++ - -fsyntax-only 2>&1 | grep ^./usr | xargs"
      OUTPUT_VARIABLE CXX_STD_INCLUDE_DIRS)
  else()
    set(CXX_STD_INCLUDE_DIRS "")
  endif()
  add_custom_target(tags
    COMMAND bash -c "ctags -R --c++-kinds=+p --fields=+iaS --extra=+q ${CMAKE_SOURCE_DIR}/src ${CXX_STD_INCLUDE_DIRS}"
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

add_subdirectory(src/gna-lib/kernels)
add_subdirectory(src/gna-lib)
add_subdirectory(src/tools)
