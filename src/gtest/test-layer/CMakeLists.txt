#INTEL CONFIDENTIAL
#Copyright 2019-2020 Intel Corporation.

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
unset(PREFIX)
set(PROJECT_NAME TEST-LAYER)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(${PROJECT_NAME})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  set(EXTRA_LIBS ${CMAKE_DL_LIBS})
  set(EXTRA_DEFS "")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(EXTRA_LIBS "Cfgmgr32.lib")
  set(EXTRA_DEFS "INTEL_GNA_DLLEXPORT=1" -DGNA_LIBRARY_VERSION_STRING=\"2.1.0.0\")
  set(gna_test_layer_compile_options )
endif()

set(EXTRA_DEFS ${GNA_LIBRARY_VER} ${EXTRA_DEFS})

set(gna_hw_module_interface_sources
  ${SRC_DIR}/HwModuleInterface.hpp
  ${SRC_DIR}/WindowsHwModuleInterface.hpp
  ${SRC_DIR}/LinuxHwModuleInterface.hpp
  ${SRC_DIR}/HwModuleInterface.cpp
  ${SRC_DIR}/WindowsHwModuleInterface.cpp
  ${SRC_DIR}/LinuxHwModuleInterface.cpp
  )

# TODO: consider making gna-api target an object library
# to avoid compiling sources twice
add_executable(test-layer
  ${gna_test_layer_sources}
  test-internal.cpp
  test-layer.cpp
  test-transpose-layer.cpp
  test-affine-layer.cpp
  test-diagonal-layer.cpp
  test-recurrent-layer.cpp
  ${COMMON_DIR}/profiler.cpp
  ${SRC_DIR}/AccelerationDetector.cpp
  ${SRC_DIR}/ActivationFunction.cpp
  ${SRC_DIR}/ActivationHelper.cpp
  ${SRC_DIR}/ActiveList.cpp
  ${SRC_DIR}/AffineFunctions.cpp
  ${SRC_DIR}/AffineLayerCapabilities.cpp
  ${SRC_DIR}/AffineLayers.cpp
  ${SRC_DIR}/ApiProperty.cpp
  ${SRC_DIR}/AuxiliaryCapabilities.cpp
  ${SRC_DIR}/Bias.cpp
  ${SRC_DIR}/BufferMap.cpp
  ${SRC_DIR}/Capabilities.cpp
  ${SRC_DIR}/CompiledModel.cpp
  ${SRC_DIR}/Component.cpp
  ${SRC_DIR}/ConvolutionalFunctions.cpp
  ${SRC_DIR}/ConvolutionalFunctions2D.cpp
  ${SRC_DIR}/ConvolutionalLayer.cpp
  ${SRC_DIR}/ConvolutionalLayer2D.cpp
  ${SRC_DIR}/ConvolutionalLayer2DCapabilities.cpp
  ${SRC_DIR}/CopyLayer.cpp
  ${SRC_DIR}/DataMode.cpp
  ${SRC_DIR}/Device.cpp
  ${SRC_DIR}/DeviceLayerSupport.cpp
  ${SRC_DIR}/DeviceManager.cpp
  ${SRC_DIR}/DriverInterface.cpp
  ${SRC_DIR}/ExternalBuffer.cpp
  ${SRC_DIR}/GmmLayer.cpp
  ${SRC_DIR}/GmmLayerCapabilities.cpp
  ${SRC_DIR}/HardwareCapabilities.cpp
  ${SRC_DIR}/HardwareLayer.cpp
  ${SRC_DIR}/HardwareModel.cpp
  ${SRC_DIR}/HardwareModelNoMMU.cpp
  ${SRC_DIR}/HardwareModelScorable.cpp
  ${SRC_DIR}/HardwareModelSue1.cpp
  ${SRC_DIR}/HardwareRequest.cpp
  ${SRC_DIR}/Layer.cpp
  ${SRC_DIR}/LayerCapabilities.cpp
  ${SRC_DIR}/LayerConfiguration.cpp
  ${SRC_DIR}/LayerDescriptor.cpp
  ${SRC_DIR}/LayerInput.cpp
  ${SRC_DIR}/LayerOutput.cpp
  ${SRC_DIR}/Layout.cpp
  ${SRC_DIR}/Logger.cpp
  ${SRC_DIR}/Memory.cpp
  ${SRC_DIR}/MemoryContainer.cpp
  ${SRC_DIR}/ModelDumper.cpp
  ${SRC_DIR}/ModelError.cpp
  ${SRC_DIR}/ModelExportConfig.cpp
  ${SRC_DIR}/ModelWrapper.cpp
  ${SRC_DIR}/OperationConfig.cpp
  ${SRC_DIR}/ParameterLimits.cpp
  ${SRC_DIR}/PoolingFunctions.cpp
  ${SRC_DIR}/PoolingFunctions2D.cpp
  ${SRC_DIR}/ProfilerConfiguration.cpp
  ${SRC_DIR}/RecurrentFunction.cpp
  ${SRC_DIR}/RecurrentLayer.cpp
  ${SRC_DIR}/RequestBuilder.cpp
  ${SRC_DIR}/RequestConfiguration.cpp
  ${SRC_DIR}/Request.cpp
  ${SRC_DIR}/RequestHandler.cpp
  ${SRC_DIR}/Scalar.cpp
  ${SRC_DIR}/Shape.cpp
  ${SRC_DIR}/SoftwareModel.cpp
  ${SRC_DIR}/StringHelper.cpp
  ${SRC_DIR}/SubModel.cpp
  ${SRC_DIR}/Tensor.cpp
  ${SRC_DIR}/ThreadPool.cpp
  ${SRC_DIR}/Transform.cpp
  ${SRC_DIR}/TransformMap.cpp
  ${SRC_DIR}/TransposeLayer.cpp
  ${SRC_DIR}/Validator.cpp
  ${SRC_DIR}/Weight.cpp
  ${SRC_DIR}/LinuxDriverInterface.cpp
  ${SRC_DIR}/WindowsDriverInterface.cpp
  ${API_IMPL_DIR}/gna2-common-impl.cpp
  ${API_IMPL_DIR}/gna2-device-impl.cpp
  ${API_IMPL_DIR}/gna2-inference-impl.cpp
  ${API_IMPL_DIR}/gna2-instrumentation-impl.cpp
  ${API_IMPL_DIR}/gna2-memory-impl.cpp
  ${API_IMPL_DIR}/gna2-model-impl.cpp
  ${API_IMPL_DIR}/gna2-model-export-impl.cpp
  ${API_IMPL_DIR}/gna2-capability-impl.cpp
  ${gna_hw_module_interface_sources})  

target_compile_definitions(test-layer
  PRIVATE ${EXTRA_DEFS} PROFILE PROFILE_DETAILED HW_VERBOSE=0)

target_link_libraries(test-layer
  gtest
  ${EXTRA_LIBS}
  ${kernel_libraries})

add_dependencies(test-layer ${kernel_libraries})

target_include_directories(test-layer
  PRIVATE ${SRC_DIR} ${COMMON_DIR} ${API_DIR} ${API_IMPL_DIR} ${KERNEL_DIR} ${GNA_HW_LIB_INCLUDE_PATH})

set_target_properties(test-layer
  PROPERTIES
  EXCLUDE_FROM_DEFAULT_BUILD_KLOCWORK TRUE
  OUTPUT_NAME test-layer)
set_property(TARGET test-layer PROPERTY FOLDER tests/gna-lib)

add_target_to_postbuild_test_run(test-layer)
