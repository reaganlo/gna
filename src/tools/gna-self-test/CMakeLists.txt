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

set(gna_self_test_sources
    gna-self-test.cpp
    HardwareSelfTest.cpp
    SelfTest.cpp
    SampleModelForGnaSelfTest.cpp)
set(gna_self_test_headers
    SelfTest.h
    MultiOsHardwareSelfTest.h
    HardwareSelfTest.h
    SampleModelForGnaSelfTest.h)
set(GNA_SELF_TEST_LINK_LIBS
  gna-api)

#gna-self-test specific part for Windows
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(gna_self_test_sources
    ${gna_self_test_sources}
    WindowsHardwareSelfTest.cpp)
  set(gna_self_test_headers
    ${gna_self_test_headers}
    WindowsHardwareSelfTest.h)
  set(GNA_SELF_TEST_LINK_LIBS
    ${GNA_SELF_TEST_LINK_LIBS}
    setupapi)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
  OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  if(NOT DEFINED LIB_PCI)
    find_library(LIB_PCI
      NAMES pci
      DOC "Path to libpci.so, may be set to false to force not using")
    message("Looking for pci library: ${LIB_PCI}")
  endif()
  if(LIB_PCI)
    set(GNA_SELF_TEST_LINK_LIBS
      ${GNA_SELF_TEST_LINK_LIBS}
      pci)
    set(gna_self_test_sources
      ${gna_self_test_sources}
      LinuxPciLibBasedImpl.cpp)
  else()
    set(gna_self_test_sources
      ${gna_self_test_sources}
      LinuxNoPciLibBasedImpl.cpp)
  endif()
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    set(ANDROID_CPP_LIB_SHARED_PATH "${CMAKE_ANDROID_STANDALONE_TOOLCHAIN}/x86_64-linux-android/lib64/libc++_shared.so")
    set(GNA_SELF_TEST_LINK_LIBS
      ${GNA_SELF_TEST_LINK_LIBS}
      ${ANDROID_CPP_LIB_SHARED_PATH})
  endif()

  set(gna_self_test_sources
    ${gna_self_test_sources}
    LinuxHardwareSelfTest.cpp
    PciDeviceInfo.cpp)
  set(gna_self_test_headers
    ${gna_self_test_headers}
    LinuxHardwareSelfTest.h
    PciDeviceInfo.h)
endif()

add_executable(gna-self-test
  ${gna_self_test_sources}
  ${gna_self_test_headers})

target_link_libraries(gna-self-test ${GNA_SELF_TEST_LINK_LIBS})
target_include_directories(gna-self-test PRIVATE ${COMMON_DIR} ${API_DIR})

set_gna_target_properties(gna-self-test)
set_gna_compile_definitions(gna-self-test)
set_gna_compile_options(gna-self-test)

copy_gna_api(gna-self-test NOT_USED)
copy_pdb_windows(gna-self-test)
strip_symbols(gna-self-test)