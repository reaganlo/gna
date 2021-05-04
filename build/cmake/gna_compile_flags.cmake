# INTEL CONFIDENTIAL
# Copyright 2020 Intel Corporation.

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

set(GNA_COMPILE_FLAGS)
set(GNA_COMPILE_ERROR_FLAGS)
set(GNA_COMPILE_FLAGS_DEBUG)
set(GNA_COMPILE_FLAGS_RELEASE)

set(GNA_COMPILE_DEFS)
set(GNA_COMPILE_DEFS_DEBUG DEBUG=1)
set(GNA_COMPILE_DEFS_RELEASE DEBUG=0)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(GNA_COMPILE_DEFS ${GNA_COMPILE_DEFS} /DWIN32 /D_WINDOWS)
  set(GNA_COMPILE_FLAGS ${GNA_COMPILE_FLAGS} /EHa /Zi /sdl)
  set(GNA_COMPILE_ERROR_FLAGS /WX)

  # Warnings
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS} /W4)
  else()
    set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS} /Wall)
  endif()

  set(GNA_COMPILE_FLAGS_DEBUG ${GNA_COMPILE_FLAGS_DEBUG} /Od /RTC1)
  set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /Oi /Gy /guard:cf)

  set(GNA_WINDOWS_RUNTIME_LINKAGE /MD)
  option(GNA_WINDOWS_RUNTIME_LINKAGE_STATIC "For UWP compliance" ON)
  if(${GNA_WINDOWS_RUNTIME_LINKAGE_STATIC})
    set(GNA_WINDOWS_RUNTIME_LINKAGE /MT)
  endif()

  # Debug/Release Multithreaded libraries
  add_compile_options(
    $<$<CONFIG:WIN_DEBUG>:${GNA_WINDOWS_RUNTIME_LINKAGE}d>
    $<$<CONFIG:WIN_RELEASE>:${GNA_WINDOWS_RUNTIME_LINKAGE}>
    $<$<CONFIG:KLOCWORK>:${GNA_WINDOWS_RUNTIME_LINKAGE}>)

  # Optimization level
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /O3)
    # Qinline-forceinline disabled due to compilation hang when IPO disabled (IPO is dusabled due to guard:cf).
    # set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /Qinline-forceinline)
    # workaround for bug https://software.intel.com/en-us/forums/intel-c-compiler/topic/798645
    set(GNA_ICL_DEBUG_WORKAROUND "/NODEFAULTLIB:\"libcpmt.lib\"")
    # remove debug_opt_report section from dll
    set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /Qopt-report-embed-)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /O2)
  endif()

  # Linker options
  set(GNA_LINKER_FLAGS "/DEBUG")
  set(GNA_LINKER_FLAGS_DEBUG "/INCREMENTAL ${GNA_ICL_DEBUG_WORKAROUND}")
  set(GNA_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO /NOLOGO /OPT:REF /OPT:ICF /guard:cf
                               /PDBSTRIPPED:$(TargetDir)$(TargetName)Public.pdb")
  # Integrity check due to security review
  option(GNA_LINKER_FLAGS_ALWAYS_RELEASE_INTEGRITYCHECK "Enables /INTEGRITYCHECK linker flag in release builds" OFF)
  option(GNA_LINKER_FLAGS_ICC_RELEASE_INTEGRITYCHECK "Enables /INTEGRITYCHECK linker flag in release builds using Intel C++ Compiler" ON)
  if(GNA_LINKER_FLAGS_ALWAYS_RELEASE_INTEGRITYCHECK
    OR (GNA_LINKER_FLAGS_ICC_RELEASE_INTEGRITYCHECK
    AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel"))
    # have to be set as can not run tests before binary image signed
    set(GNA_DISABLE_TEST_RUN $<NOT:$<CONFIG:WIN_DEBUG>>)
    set(GNA_LINKER_FLAGS_RELEASE "${GNA_LINKER_FLAGS_RELEASE} /INTEGRITYCHECK")
  endif()
    # remove Debug Information from image (coffgrpinfo section)
    set(GNA_LINKER_FLAGS_RELEASE "${GNA_LINKER_FLAGS_RELEASE} /nocoffgrpinfo")
    set(GNA_LINKER_FLAGS_RELEASE "${GNA_LINKER_FLAGS_RELEASE} /novcfeature")
    # consider replacing(nocoffgrpinfo, novcfeature) with the folowing (no pdb will be created)
    # set(GNA_LINKER_FLAGS_RELEASE "${GNA_LINKER_FLAGS_RELEASE} /DEBUG:NONE")
else()
  set(GNA_COMPILE_DEFS_RELEASE ${GNA_COMPILE_DEFS_RELEASE} _FORTIFY_SOURCE=2)

  # All compilers warnings
  set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS}
      -Wall -Werror
      -Wextra -Wshadow -Wunused -Wformat)

  # GCC & Clang warnings
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"
      OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS}
        -Wpedantic -Wconversion -Wdouble-promotion)

    # Clang double braces bug: https://bugs.llvm.org/show_bug.cgi?id=21629
    set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS} -Wno-missing-braces)
  endif()

  # GCC only warnings
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # -Wuseless-cast not applicable - _mm256_i32gather_epi32 implemetation generates warning
    set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS} -Wlogical-op)
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "6.0")
      set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS}
          -Wnull-dereference -Wduplicated-cond)
    endif()
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "7.0")
      set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS}
          -Wduplicated-branches)
    endif()
  endif()

  # Clang only warnings
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS} -Wno-\#pragma-messages)
  endif()

  # Optimization and symbols
  set(GNA_COMPILE_FLAGS_DEBUG ${GNA_COMPILE_FLAGS_DEBUG}
      -g -O0)
  set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE}
      -fvisibility=hidden -fstack-protector-all -O3)

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # ICC intrinsics inline expansion
    set(GNA_COMPILE_FLAGS ${GNA_COMPILE_FLAGS} -fbuiltin)
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
      # Don't use Intel shared libraries
      set(GNA_COMMON_SHARED_LINKER_FLAGS "-static-intel")
    endif()
  endif()

  # linker security and optimization flags
  # relro - Hardening ELF binaries using Relocation Read-Only
  set(GNA_LINKER_FLAGS "-z now")
  set(GNA_LINKER_FLAGS_RELEASE "-fdata-sections -ffunction-sections -Wl,--gc-sections -z relro")

  set(GNA_CC_COMPILE_FLAGS ${GNA_COMPILE_FLAGS})
  set(GNA_COMPILE_ERROR_FLAGS ${GNA_COMPILE_ERROR_FLAGS}
      -Woverloaded-virtual -Wnon-virtual-dtor)
  set(GNA_COMPILE_COMMON_FLAGS ${GNA_COMPILE_FLAGS})
  set(GNA_COMPILE_FLAGS ${GNA_COMPILE_FLAGS} ${GNA_COMPILE_ERROR_FLAGS})

  set(GNA_CC_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE})
  set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE}
      -fvisibility-inlines-hidden)

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
      message(WARNING "IPO disabled as causes compiler failure with guard:cf.")
      #       error : ** The compiler has encountered an unexpected problem.
      #       ** Segmentation violation signal raised. **
      #       Access violation or stack overflow. Please contact Intel Support for assistance.
      #       xilink: : error #10014: problem during multi-file optimization compilation (code 4)
      # set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /Qipo)
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
      set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} /GL)
    endif()
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} -ipo)
    elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} -flto=thin)
    else()
      if(${TARGET_OS} STREQUAL "ChromeOS")
	set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE})
      else()
	set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} -flto -fno-fat-lto-objects)
      endif()
    endif()
  elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      # Intel compiler hangs or crashes when using -ipo option for Android target
      # icpc: error #10014: problem during multi-file optimization compilation (code 4)
      set(GNA_COMPILE_FLAGS_RELEASE ${GNA_COMPILE_FLAGS_RELEASE} -flto -fno-fat-lto-objects)
    endif()
  endif()
endif()

# set 32-bit compilation flags
if(CMAKE_ARCHITECTURE STREQUAL x86)
  include(${CMAKE_SOURCE_DIR}/build/cmake/set_x86_flags.cmake)
endif()

set(CMAKE_SHARED_LINKER_FLAGS                      "${CMAKE_SHARED_LINKER_FLAGS} ${GNA_LINKER_FLAGS} ${GNA_COMMON_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}_DEBUG   "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${GNA_LINKER_FLAGS_DEBUG}  ${GNA_COMMON_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_${OS_PREFIX}_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${GNA_LINKER_FLAGS_RELEASE} ${GNA_COMMON_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_KLOCWORK             "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${GNA_LINKER_FLAGS_RELEASE} ${GNA_COMMON_SHARED_LINKER_FLAGS}")

set(CMAKE_EXE_LINKER_FLAGS                         "${CMAKE_EXE_LINKER_FLAGS} ${GNA_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}_DEBUG      "${CMAKE_EXE_LINKER_DEBUG} ${GNA_LINKER_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS_${OS_PREFIX}_RELEASE    "${CMAKE_EXE_LINKER_RELEASE} ${GNA_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_KLOCWORK                "${CMAKE_EXE_LINKER_RELEASE} ${GNA_LINKER_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS_${OS_PREFIX}_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_${OS_PREFIX}_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_KLOCWORK ${CMAKE_CXX_FLAGS_RELEASE})

set(CMAKE_C_FLAGS_${OS_PREFIX}_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_${OS_PREFIX}_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_KLOCWORK ${CMAKE_C_FLAGS_RELEASE})
