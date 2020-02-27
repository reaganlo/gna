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

function (strip_symbols TARG_NAME)
  set(GNA_TARG_RELEASE_OUT_DIR ${GNA_TOOLS_RELEASE_OUT_DIR}/${TARG_NAME}/${CMAKE_ARCHITECTURE})
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    add_custom_command(TARGET ${TARG_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        -DPDB_PUBLIC=${GNA_TARG_RELEASE_OUT_DIR}/${TARG_NAME}Public.pdb
        -DPDB_PATH=${GNA_TARG_RELEASE_OUT_DIR}
        -DFILE_NAME=${TARG_NAME}.pdb
        -P ${CMAKE_SOURCE_DIR}/build/cmake/pdb_public.cmake)
  endif()

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_BUILD_TYPE} STREQUAL "LNX_RELEASE")
      add_custom_command(TARGET ${TARG_NAME} POST_BUILD
        COMMAND cp $<TARGET_FILE:${TARG_NAME}> $<TARGET_FILE:${TARG_NAME}>.dbg
        COMMAND strip --only-keep-debug $<TARGET_FILE:${TARG_NAME}>.dbg
        COMMAND strip --strip-unneeded $<TARGET_FILE:${TARG_NAME}>)
    endif()
  endif()
endfunction(strip_symbols)

function (copy_pdb_windows TARG_NAME)
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    add_custom_command(TARGET ${TARG_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PDB_FILE:gna-api>
        $<TARGET_FILE_DIR:${TARG_NAME}>)
  endif()
endfunction(copy_pdb_windows)

function (copy_gna_api DST_TARGET OUT_NEW_TARGET)
  set(NEW_TARGET "copy-gna-api-to-${DST_TARGET}")
  set(${OUT_NEW_TARGET} "${NEW_TARGET}" PARENT_SCOPE)
  add_custom_target(${NEW_TARGET} ALL
    COMMENT "Running target: ${NEW_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    $<TARGET_FILE_DIR:gna-api>
    $<TARGET_FILE_DIR:${DST_TARGET}>)
  set_target_properties(${NEW_TARGET}
    PROPERTIES
    FOLDER tools/${DST_TARGET})
endfunction(copy_gna_api)

function (set_gna_compile_options TARG_NAME)
  target_compile_options(${TARG_NAME}
    PRIVATE
    ${GNA_COMPILE_FLAGS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
    $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>
    ${EXTRA_EXE_COMPILE_OPTIONS})
endfunction(set_gna_compile_options)

function (set_gna_compile_definitions TARG_NAME)
  target_compile_definitions(${TARG_NAME}
    PRIVATE
    ${GNA_COMPILE_DEFS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
    $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>)
endfunction(set_gna_compile_definitions)

function (set_gna_target_properties TARG_NAME)
  set_target_properties(${TARG_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_TOOLS_DEBUG_OUT_DIR}/${TARG_NAME}/${CMAKE_ARCHITECTURE}
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_TOOLS_RELEASE_OUT_DIR}/${TARG_NAME}/${CMAKE_ARCHITECTURE}
    EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
    OUTPUT_NAME ${TARG_NAME}
    FOLDER tools/${TARG_NAME})
endfunction(set_gna_target_properties)