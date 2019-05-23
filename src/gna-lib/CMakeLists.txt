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
set(PROJECT_NAME gna-api)
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT DEFINED GNA_VERSION_MAJOR)
set(GNA_VERSION_MAJOR 2)
endif()

if(NOT DEFINED GNA_VERSION_MINOR)
set(GNA_VERSION_MINOR 0)
endif()

if(NOT DEFINED GNA_VERSION_PATCH)
set(GNA_VERSION_PATCH 0)
endif()

if(NOT DEFINED GNA_VERSION_RELEASE)
set(GNA_VERSION_RELEASE 0)
endif()

if(NOT DEFINED GNA_VERSION)
set(GNA_VERSION ${GNA_VERSION_MAJOR}.${GNA_VERSION_MINOR}.${GNA_VERSION_PATCH}.${GNA_VERSION_RELEASE})
endif()

project(${PROJECT_NAME})

set(IOCTL_SENDER_PREFIX ${CMAKE_SYSTEM_NAME})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(API_EXTRA_LIBS "")
  set(API_EXTRA_DEFS "")
  set(IOCTL_SENDER_PREFIX "Linux")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(API_EXTRA_LIBS "setupapi.lib")
  set(API_EXTRA_DEFS "-DINTEL_GNA_DLLEXPORT=1")
endif()

set(GNA_API_INCL_DIR ${GNA_BINARY_DIR}/gna-lib/include)
set(GNA_LIB_OUT_DIR ${GNA_BINARY_DIR}/gna-lib)
set(GNA_LIB_DEBUG_OUT_DIR ${GNA_LIB_OUT_DIR}/${OS_PREFIX}-DEBUG/${CMAKE_ARCHITECTURE})
set(GNA_LIB_RELEASE_OUT_DIR ${GNA_LIB_OUT_DIR}/${OS_PREFIX}-RELEASE/${CMAKE_ARCHITECTURE})

set(GNA_API_VERBOSE_INCL_DIR ${GNA_BINARY_DIR}/gna-lib-verbose/include)
set(GNA_LIB_VERBOSE_OUT_DIR ${GNA_BINARY_DIR}/gna-lib-verbose)
set(GNA_LIB_VERBOSE_DEBUG_OUT_DIR ${GNA_LIB_VERBOSE_OUT_DIR}/${OS_PREFIX}-DEBUG/${CMAKE_ARCHITECTURE})
set(GNA_LIB_VERBOSE_RELEASE_OUT_DIR ${GNA_LIB_VERBOSE_OUT_DIR}/${OS_PREFIX}-RELEASE/${CMAKE_ARCHITECTURE})

set(GNA_API_PDB_STRIPPED ${GNA_LIB_RELEASE_OUT_DIR}/gnaPublic.pdb)
set(GNA_API_VERBOSE_PDB_STRIPPED ${GNA_LIB_VERBOSE_RELEASE_OUT_DIR}/gna-verbosePublic.pdb)

set(gna_lib_sources
  ${SRC_DIR}/api.cpp
  ${SRC_DIR}/AccelerationDetector.cpp
  ${SRC_DIR}/ActivationFunction.cpp
  ${SRC_DIR}/ActiveList.cpp
  ${SRC_DIR}/AffineFunctions.cpp
  ${SRC_DIR}/AffineLayers.cpp
  ${SRC_DIR}/ApiProperty.cpp
  ${SRC_DIR}/Bias.cpp
  ${SRC_DIR}/Capabilities.cpp
  ${SRC_DIR}/CompiledModel.cpp
  ${SRC_DIR}/Component.cpp
  ${SRC_DIR}/ConvolutionalFunctions.cpp
  ${SRC_DIR}/ConvolutionalFunctions2D.cpp
  ${SRC_DIR}/ConvolutionalLayer.cpp
  ${SRC_DIR}/ConvolutionalLayer2D.cpp
  ${SRC_DIR}/CopyLayer.cpp
  ${SRC_DIR}/Cnn2DuArch.cpp
  ${SRC_DIR}/DataMode.cpp
  ${SRC_DIR}/Device.cpp
  ${SRC_DIR}/DeviceLayerSupport.cpp
  ${SRC_DIR}/DeviceManager.cpp
  ${SRC_DIR}/GmmLayer.cpp
  ${SRC_DIR}/HardwareCapabilities.cpp
  ${SRC_DIR}/HardwareLayer.cpp
  ${SRC_DIR}/HardwareModel.cpp
  ${SRC_DIR}/HardwareModelScorable.cpp
  ${SRC_DIR}/HardwareModelSue1.cpp
  ${SRC_DIR}/HardwareRequest.cpp
  ${SRC_DIR}/Layer.cpp
  ${SRC_DIR}/LayerConfiguration.cpp
  ${SRC_DIR}/LayerDescriptor.cpp
  ${SRC_DIR}/LayerInput.cpp
  ${SRC_DIR}/LayerOutput.cpp
  ${SRC_DIR}/Layout.cpp
  ${SRC_DIR}/Logger.cpp
  ${SRC_DIR}/Memory.cpp
  ${SRC_DIR}/ModelDumper.cpp
  ${SRC_DIR}/ModelExportConfig.cpp
  ${SRC_DIR}/ModelWrapper.cpp
  ${SRC_DIR}/OperationConfig.cpp
  ${SRC_DIR}/PoolingFunctions.cpp
  ${SRC_DIR}/PoolingFunctions2D.cpp
  ${SRC_DIR}/RecurrentLayer.cpp
  ${SRC_DIR}/RequestBuilder.cpp
  ${SRC_DIR}/RequestConfiguration.cpp
  ${SRC_DIR}/Request.cpp
  ${SRC_DIR}/RequestHandler.cpp
  ${SRC_DIR}/Scalar.cpp
  ${SRC_DIR}/Shape.cpp
  ${SRC_DIR}/SoftwareModel.cpp
  ${SRC_DIR}/SubModel.cpp
  ${SRC_DIR}/Tensor.cpp
  ${SRC_DIR}/ThreadPool.cpp
  ${SRC_DIR}/Transform.cpp
  ${SRC_DIR}/TransformMap.cpp
  ${SRC_DIR}/TransposeLayer.cpp
  ${SRC_DIR}/Validator.cpp
  ${SRC_DIR}/Weight.cpp
  ${SRC_DIR}/${IOCTL_SENDER_PREFIX}DriverInterface.cpp)

set(gna_api_impl_src_2
  ${API_IMPL_DIR}/gna2-common-impl.cpp
  ${API_IMPL_DIR}/gna2-device-impl.cpp
  ${API_IMPL_DIR}/gna2-inference-impl.cpp
  ${API_IMPL_DIR}/gna2-instrumentation-impl.cpp
  ${API_IMPL_DIR}/gna2-memory-impl.cpp
  ${API_IMPL_DIR}/gna2-model-impl.cpp
  ${API_IMPL_DIR}/gna2-model-export-impl.cpp
  ${API_IMPL_DIR}/gna2-capability-impl.cpp)

set(gna_lib_sources
  ${gna_lib_sources} ${gna_api_impl_src_2})

set(gna_lib_headers
  ${SRC_DIR}/AccelerationDetector.h
  ${SRC_DIR}/ActivationFunction.h
  ${SRC_DIR}/ActiveList.h
  ${SRC_DIR}/AffineFunctions.h
  ${SRC_DIR}/ApiProperty.h
  ${SRC_DIR}/ApiWrapper.h
  ${SRC_DIR}/Capabilities.h
  ${SRC_DIR}/CompiledModel.h
  ${SRC_DIR}/Component.h
  ${SRC_DIR}/ConvolutionalFunctions.h
  ${SRC_DIR}/ConvolutionalFunctions2D.h
  ${SRC_DIR}/ConvolutionalLayer.h
  ${SRC_DIR}/ConvolutionalLayer2D.h
  ${SRC_DIR}/CopyLayer.h
  ${SRC_DIR}/Cnn2DuArch.h
  ${SRC_DIR}/DataMode.h
  ${SRC_DIR}/Device.h
  ${SRC_DIR}/DeviceLayerSupport.h
  ${SRC_DIR}/DeviceManager.h
  ${SRC_DIR}/DriverInterface.h
  ${SRC_DIR}/Expect.h
  ${SRC_DIR}/GmmLayer.h
  ${SRC_DIR}/GnaTypes.h
  ${SRC_DIR}/HardwareCapabilities.h
  ${SRC_DIR}/HardwareLayer.h
  ${SRC_DIR}/HardwareModel.h
  ${SRC_DIR}/HardwareModelScorable.h
  ${SRC_DIR}/HardwareModelSue1.h
  ${SRC_DIR}/HardwareRequest.h
  ${SRC_DIR}/IScorable.h
  ${SRC_DIR}/LayerConfiguration.h
  ${SRC_DIR}/Layer.h
  ${SRC_DIR}/LayerDescriptor.h
  ${SRC_DIR}/LayerInput.h
  ${SRC_DIR}/LayerOutput.h
  ${SRC_DIR}/Layout.h
  ${SRC_DIR}/Logger.h
  ${SRC_DIR}/Memory.h
  ${SRC_DIR}/ModelExportConfig.h
  ${SRC_DIR}/ModelWrapper.h
  ${SRC_DIR}/OperationConfig.h
  ${SRC_DIR}/ParameterLimits.h
  ${SRC_DIR}/PoolingFunctions.h
  ${SRC_DIR}/PoolingFunctions2D.h
  ${SRC_DIR}/RecurrentLayer.h
  ${SRC_DIR}/RequestBuilder.h
  ${SRC_DIR}/RequestConfiguration.h
  ${SRC_DIR}/Request.h
  ${SRC_DIR}/RequestHandler.h
  ${SRC_DIR}/Scalar.h
  ${SRC_DIR}/Shape.h
  ${SRC_DIR}/SoftwareModel.h
  ${SRC_DIR}/SubModel.h
  ${SRC_DIR}/Tensor.h
  ${SRC_DIR}/ThreadPool.h
  ${SRC_DIR}/Transform.h
  ${SRC_DIR}/TransformMap.h
  ${SRC_DIR}/TransposeLayer.h
  ${SRC_DIR}/Validator.h
  ${SRC_DIR}/Weight.h
  ${SRC_DIR}/${IOCTL_SENDER_PREFIX}DriverInterface.h)

set(gna_api_impl_headers_2
  ${API_IMPL_DIR}/gna2-impl.h
  ${API_IMPL_DIR}/gna2-capability-impl.h
  ${API_IMPL_DIR}/gna2-common-impl.h
  ${API_IMPL_DIR}/gna2-device-impl.h
  ${API_IMPL_DIR}/gna2-inference-impl.h
  ${API_IMPL_DIR}/gna2-instrumentation-impl.h
  ${API_IMPL_DIR}/gna2-memory-impl.h
  ${API_IMPL_DIR}/gna2-model-impl.h
  ${API_IMPL_DIR}/gna2-model-export-impl.h
  ${API_IMPL_DIR}/gna2-validation-impl.h)

set(gna_lib_headers
  ${gna_lib_headers} ${gna_api_impl_headers_2})

set(gna_api_headers
  ${COMMON_DIR}/gna-api.h
  ${COMMON_DIR}/gna-api-types-xnn.h
  ${COMMON_DIR}/gna-api-types-gmm.h
  ${COMMON_DIR}/gna-api-status.h)

set(gna_api_extra
  ${COMMON_DIR}/gna-api-dumper.h
  ${COMMON_DIR}/gna-api-instrumentation.h)

set(gna_api_headers_2
  ${API_DIR}/gna2-api.h
  ${API_DIR}/gna2-common-api.h
  ${API_DIR}/gna2-device-api.h
  ${API_DIR}/gna2-inference-api.h
  ${API_DIR}/gna2-instrumentation-api.h
  ${API_DIR}/gna2-memory-api.h
  ${API_DIR}/gna2-model-api.h)

set(gna_api_headers_extra_2
  ${API_DIR}/gna2-model-export-api.h
  ${API_DIR}/gna2-model-suecreek-header.h
  ${API_DIR}/gna2-capability-api.h)

set(common_sources
  ${COMMON_DIR}/Address.h
  ${COMMON_DIR}/common.h
  ${COMMON_DIR}/GnaException.h
  ${COMMON_DIR}/Macros.h
  ${COMMON_DIR}/profiler.cpp
  ${COMMON_DIR}/profiler.h)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set_source_files_properties(${COMMON_DIR}/version.rc PROPERTIES LANGUAGE RC)
  set(gna_rc_files
    ${COMMON_DIR}/resource.h
    ${COMMON_DIR}/version.rc)
  set(windows_ddi
    ${COMMON_DIR}/GnaDrvApi.h
    ${COMMON_DIR}/GnaDrvApiWin.h
    ${COMMON_DIR}/GnaDrvApiWinDebug.h)
  list(APPEND common_sources ${gna_rc_files} ${windows_ddi})
endif()

set(api_headers ${gna_api_headers} ${gna_api_extra})
source_group(Api FILES ${api_headers})

set(api_headers_2 ${gna_api_headers_2} ${gna_api_headers_extra_2})
source_group(Api2 FILES ${api_headers_2})

source_group("Api2 Source Files" FILES ${gna_api_impl_src_2})

set(api_headers ${api_headers} ${api_headers_2})

add_library(gna-api SHARED)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(gna_lib_headers ${gna_lib_headers} ${COMMON_DIR}/gna.h)
endif()

target_include_directories(gna-api
  PRIVATE ${SRC_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR})

target_include_directories(gna-api PUBLIC
  $<BUILD_INTERFACE:${COMMON_DIR}>
  $<INSTALL_INTERFACE:include/gna${GNA_VERSION_MAJOR}>)

target_sources(gna-api
  PRIVATE ${common_sources} ${gna_lib_sources} ${gna_lib_headers} ${kernel_api_headers})

target_sources(gna-api PUBLIC
  $<BUILD_INTERFACE:${api_headers}>)

set_target_properties(gna-api
  PROPERTIES
  FOLDER library
  VERSION ${GNA_VERSION}
  SOVERSION ${GNA_VERSION_MAJOR}
  LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_DEBUG_OUT_DIR}
  LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_RELEASE_OUT_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_DEBUG_OUT_DIR}
  ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_RELEASE_OUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_DEBUG_OUT_DIR}
  RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_RELEASE_OUT_DIR}
  VS_USER_PROPS ${CMAKE_SOURCE_DIR}/build/common/version-dll.props
  OUTPUT_NAME gna)

target_compile_definitions(gna-api
  PRIVATE
  ${GNA_COMPILE_DEFS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
  ${API_EXTRA_DEFS}
  -DPROFILE -DPROFILE_DETAILED -DHW_VERBOSE=0)

target_compile_options(gna-api
  PRIVATE
  ${GNA_COMPILE_FLAGS}
  $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
  $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
  $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>)

target_link_libraries(gna-api
  PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${API_EXTRA_LIBS} ${kernel_libraries})

add_dependencies(gna-api ${kernel_libraries})

file(MAKE_DIRECTORY ${GNA_API_INCL_DIR})
add_custom_command(TARGET gna-api POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${api_headers}
  ${GNA_API_INCL_DIR})

add_custom_command(TARGET gna-api POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${api_headers}
  ${CMAKE_SOURCE_DIR}/samples/gna-lib/include)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  add_custom_command(TARGET gna-api POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:gna-api>
    $<TARGET_LINKER_FILE:gna-api>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:$<TARGET_FILE_DIR:gna-api>/gnaPublic.pdb>
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:$<TARGET_PDB_FILE:gna-api>>
    ${CMAKE_SOURCE_DIR}/samples/gna-lib/${CMAKE_ARCHITECTURE}
    COMMAND ${CMAKE_COMMAND}
      -DPDB_PUBLIC=${GNA_API_PDB_STRIPPED}
      -DPDB_PATH=${GNA_LIB_RELEASE_OUT_DIR}
      -DFILE_NAME=gna.pdb
      -P ${CMAKE_SOURCE_DIR}/build/cmake/pdb_public.cmake)
else()
  add_custom_command(TARGET gna-api POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_SONAME_FILE:gna-api>
    ${CMAKE_SOURCE_DIR}/samples/gna-lib/${CMAKE_ARCHITECTURE})
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  add_custom_command(TARGET gna-api POST_BUILD
    COMMAND
    echo $(API_DLL) version: $(VERSION_QB_A) ($(VERSION_FILE_A)) > $(OutDir)$(API_DLL).ver)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")

  set(verbose_sources
    ${SRC_DIR}/CompiledModelVerbose.h
    ${SRC_DIR}/DeviceVerbose.h
    ${SRC_DIR}/HardwareModelVerbose.h
    ${SRC_DIR}/HardwareModelVerbose.cpp
    ${SRC_DIR}/MemoryVerbose.h
    ${SRC_DIR}/api-verbose.cpp
    ${API_IMPL_DIR}/gna2-validation-impl.cpp)

  source_group("Verbose Source Files" FILES ${verbose_sources})

  set(api_headers_verbose_only
    ${COMMON_DIR}/gna-api-verbose.h
    ${API_DIR}/gna2-validation-api.h)

  set(api_headers_verbose
    ${api_headers}
    ${api_headers_verbose_only})

  source_group("Api2 Verbose" FILES ${api_headers_verbose_only})


  add_library(gna-api-verbose SHARED)

  target_include_directories(gna-api-verbose
    PRIVATE ${SRC_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR}
    PUBLIC ${COMMON_DIR})

  target_sources(gna-api-verbose
    PRIVATE ${common_sources} ${gna_lib_sources} ${gna_lib_headers} ${api_headers_verbose} ${kernel_api_headers} ${verbose_sources}
    PUBLIC ${api_headers_verbose})

  set_target_properties(gna-api-verbose
    PROPERTIES
    FOLDER library
    LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_VERBOSE_DEBUG_OUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_VERBOSE_RELEASE_OUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_VERBOSE_DEBUG_OUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_VERBOSE_RELEASE_OUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_DEBUG ${GNA_LIB_VERBOSE_DEBUG_OUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_${OS_PREFIX}_RELEASE ${GNA_LIB_VERBOSE_RELEASE_OUT_DIR}
    VS_USER_PROPS ${CMAKE_SOURCE_DIR}/build/common/version-dll.props
    EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
    OUTPUT_NAME gna-verbose)

  set_property(TARGET gna-api-verbose PROPERTY COMPILE_FLAGS "/WX")

  target_compile_definitions(gna-api-verbose
    PRIVATE
    ${GNA_COMPILE_DEFS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_DEFS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_DEFS_RELEASE}>
    $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_DEFS_RELEASE}>
    ${API_EXTRA_DEFS} -DPROFILE -DPROFILE_DETAILED -DHW_VERBOSE=1 -DDUMP_ENABLED)

  target_compile_options(gna-api-verbose
    PRIVATE
    ${GNA_COMPILE_FLAGS}
    $<$<CONFIG:${OS_PREFIX}_DEBUG>:${GNA_COMPILE_FLAGS_DEBUG}>
    $<$<CONFIG:${OS_PREFIX}_RELEASE>:${GNA_COMPILE_FLAGS_RELEASE}>
    $<$<CONFIG:KLOCWORK>:${GNA_COMPILE_FLAGS_RELEASE}>)

  target_link_libraries(gna-api-verbose
    PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${API_EXTRA_LIBS} ${kernel_libraries})

  add_dependencies(gna-api-verbose ${kernel_libraries})

  file(MAKE_DIRECTORY ${GNA_API_VERBOSE_INCL_DIR})
  add_custom_command(TARGET gna-api POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${api_headers_verbose}
    ${GNA_API_VERBOSE_INCL_DIR})

  add_custom_command(TARGET gna-api-verbose POST_BUILD
    COMMAND
    echo $(API_DLL) version: $(VERSION_QB_A) ($(VERSION_FILE_A)) > $(OutDir)$(API_DLL).ver
    COMMAND ${CMAKE_COMMAND}
      -DPDB_PUBLIC=${GNA_API_VERBOSE_PDB_STRIPPED}
      -DPDB_PATH=${GNA_LIB_VERBOSE_RELEASE_OUT_DIR}
      -DFILE_NAME=gna-verbose.pdb
      -P ${CMAKE_SOURCE_DIR}/build/cmake/pdb_public.cmake)
endif()

#strip symbols on linux
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  if(${CMAKE_BUILD_TYPE} STREQUAL "LNX_RELEASE")
    add_custom_command(TARGET gna-api POST_BUILD
      COMMAND cp $<TARGET_FILE:gna-api> $<TARGET_FILE:gna-api>.dbg
      COMMAND strip --only-keep-debug $<TARGET_FILE:gna-api>.dbg
      COMMAND strip --strip-unneeded $<TARGET_FILE:gna-api>)
  endif()
endif()

install(TARGETS gna-api EXPORT GnaTargets
  CONFIGURATIONS ${OS_PREFIX}_DEBUG ${OS_PREFIX}_RELEASE
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include)

install(FILES ${api_headers}
  DESTINATION include/gna${GNA_VERSION_MAJOR})

# package version
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/gna-config-version.cmake
  VERSION ${GNA_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/gna-config-version.cmake
  DESTINATION lib/cmake/gna${GNA_VERSION_MAJOR})

# build directory export
export(TARGETS gna-api
  FILE GnaTargetsConfig.cmake)

# system wide export
install(EXPORT GnaTargets
  FILE gna-config.cmake
  NAMESPACE Gna::
  DESTINATION lib/cmake/gna${GNA_VERSION_MAJOR})
