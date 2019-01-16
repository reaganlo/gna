/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.
    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/******************************************************************************
 *
 * GNA 3.0 API
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * API Definition
 *
 *****************************************************************************/

#ifndef __GNA_API_H
#define __GNA_API_H

#include <stdint.h>

#if !defined(_WIN32)
#include <assert.h>
#endif

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Library API import/export macros */
#if !defined(GNAAPI)
#    if 1 == _WIN32
#       if 1 == INTEL_GNA_DLLEXPORT
#           define GNAAPI __declspec(dllexport)
#       else
#           define GNAAPI __declspec(dllimport)
#       endif
#    else
#        if __GNUC__ >= 4
#           define GNAAPI __attribute__ ((visibility ("default")))
#        else
#           define GNAAPI
#        endif
#    endif
#endif

/******************  GNA Device API ******************/

/** GNA Device identifier **/
typedef uint32_t gna_device_id;

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////    start of changes            ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

// TODO: GNA 3.0/1 reorganize API into headers to facilitate different/separate modules
/******************************************************************************
 * Draft:
 * Model API (gna-api-model.h)
 *   - gna-api-types-xnn.h
 *   - gna-api-types-gmm (consolidate with xnn)
 *   - ANNA enhancements - non-neural-network) (only for ANNA converter)
 *   - Memory & model api
 *      - alloc/free
 *      - create
 *      - import
 *      - export (gna-api-dumper/export.h)
 * Common API (gna-api.h)
 *   - gna-api-status.h
 *   - capability api?
 *      - requires some model enums
 *      - requires some instrumentation enums
 *   - device api
 *      - device open
 *      - device close
 *   - helpers/other
  * Runtime Inference API (gna-api-inference.h)
 *   - request configs
 *   - request enqueue/wait
 *   - include/incorporate gna-api-instrumentation.h
 *****************************************************************************/



/******************************************************************************
 * GNA Capabilities API
 *****************************************************************************/
typedef enum _api_version
{
    GNA_API_NOT_SUPPORTED = GNA_NOT_SUPPORTED,
    GNA_API_1_0,
    GNA_API_2_0,
    //GNA_API_2_1_S, // TODO: use GNA_API_3_0
    GNA_API_3_0,

    GNA_API_VERSION_COUNT
} gna_api_version;

// TODO:3: rename to generation or remove and use driver device types
typedef enum _device_version
{
    GNA_DEVICE_NOT_SUPPORTED = GNA_NOT_SUPPORTED,
    GNA_0_9,                // GNA 0.9 Device Cannon Lake (CNL), no CNN support
    GNA_1_0,                // GNA 1.0 Device Gemini Lake (GLK), full featured GNA 1.0
                            // GNA 1.0 Device Ice Lake (ICL), same function set as GLK
    GNA_1_0_EMBEDDED,       // GNA 1.0 Embedded Sue Creek (SUE)
    GNA_2_0,                // GNA 2.0 Device Tiger Lake, full featured GNA 2.0 (TGL)
    GNA_2_1_EMBEDDED,       // GNA 2.1 Embedded Jelly Fish (JFL)
    GNA_3_0,                // GNA 3.0 Device Alder Lake, full featured GNA 3.0 (ADL)
    GNA_3_0_EMBEDDED,       // GNA 3.0 Embedded on Alder Lake PCH/ACE
    GNA_3_1_AUTONOMUS,      // GNA 3.1 ANNA Autonomous Embedded on Alder Lake PCH/ACE
    GMM_DEVICE,
    GNA_DEVICE_COUNT
} gna_device_generation;

/**
 *  Enumeration of device flavors
 */
typedef enum _gna_device_version
{
    GNA_UNSUPPORTED = 0x0000,   // No supported device available
    GNA_SKL     = 0x1911,   // GMM Device Sky Lake
    GNA_KBL     = 0x1911,   // GMM Device Kaby Lake // TODO:3: check KBL HW ID
    GNA_CNL     = 0x5A11,   // GNA 0.9 Device Cannon Lake, no CNN support
    GNA_GLK     = 0x3190,   // GNA 1.0 Device Gemini Lake, full featured GNA 1.0
    GNA_EHL     = 0x4511,   // GNA 1.0 Device Elkhartlake, same function set as GLK
    GNA_ICL     = 0x8A11,   // GNA 1.0 Device Ice Lake, same function set as GLK
    GNA_TGL     = 0x9A11,   // GNA 2.0 Device Tiger Lake, full featured GNA 2.0
    GNA_ADL     = 0x46AD,   // GNA 3.0 Device Alder Lake, full featured GNA 3.0
    GNA_SUE_CREEK   = 0xFFFF1,  // GNA 1.0 Embedded Sue Creek
    GNA_JELLYFISH   = 0xFFFF2,  // GNA 2.1 Embedded Jelly Fish
    GNA_ACE_EMBEDDED= 0xFFFF3,  // GNA 3.0 Embedded on Alder Lake PCH/ACE
    GNA_ACE_ANNA    = 0xFFFF4,  // GNA 3.1 ANNA Autonomous Embedded on Alder Lake PCH/ACE

} gna_device_version;

//
//// Binary flags
//typedef enum _memory_mode
//{
//    GNA_MEMORY_NOT_SUPPORTED = GNA_NOT_SUPPORTED,
//    GNA_MEMORY_MAPPED = 1,
//    GNA_MEMORY_DIRECT = 2,
//    GNA_MEMORY_DEDICATED = 4, // Server device built-in memory
//    GNA_MEMORY_FANCY_MODE = 8, // ACE extension?
//} gna_memory_mode;
//
//// helper to identify returned property type
//typedef enum _property_type
//{
//    GNA_TYPE_NOT_SUPPORTED = GNA_NOT_SUPPORTED, // not supported
//    GNA_UINT8_T,                                // cast to uint8_t
//    GNA_UINT16_T,                               // cast to uint16_t
//    GNA_UINT32_T,                               // cast to uint32_t
//    GNA_UINT64_T,                               // cast to uint64_t
//    GNA_BOOL_T,                                 // cast to bool
//
//    GNA_API_VERSION_T,                          // cast to gna_api_version
//
//    GNA_DEVICE_GENERATION_T,                       // cast to gna_device_generation
//    GNA_MEMORY_MODE_T,                          // Binary flags of gna_memory_mode
//    GNA_DATA_MODE_T,                            // Binary flags of gna_data_mode
//    GNA_BIAS_MODE_T,                            // Binary flags of gna_bias_mode
//    GNA_POOLING_MODE_T,                         // Binary flags of gna_pooling_mode
//    GNA_TENSOR_ORDER_T,                         // cast to gna_tensor_order_t
//    // can facilitate some fancy structure up to 8B
//} gna_property_type;
//
//typedef enum _api_property
//{
//    GNA_API_VERSION,                            // GNA_API_VERSION_T
//    GNA_API_BUILD,                              // GNA_UINT32_T
//    GNA_API_THREAD_COUNT,                       // GNA_UINT32_T
//    GNA_API_THREAD_COUNT_MAX,                   // GNA_UINT32_T
//
//    GNA_API_PROPERTY_COUNT
//} gna_api_property;
//
//typedef enum _device_property
//{
//    // properties for GNA_DEVICE subject
//    GNA_DEVICE_AVAILABLE_COUNT,                 // GNA_UINT32_T
//    GNA_DEVICE_ACTIVE_COUNT,                    // GNA_UINT32_T
//    GNA_DEVICE_ACTIVE_COUNT_MAX,                // GNA_UINT32_T
//    GNA_DEVICE_PROFILE,                         // GNA_DEVICE_PROFILE_T
//    GNA_DEVICE_VERSION,                         // GNA_DEVICE_GENERATION_T
//    GNA_DEVICE_DRIVER_BUILD,                    // GNA_UINT32_T
//    GNA_DEVICE_CLOCK_FREQUENCY,                 // GNA_UINT32_T
//    GNA_DEVICE_COMPUTE_ENGINE_COUNT,            // GNA_UINT32_T
//    GNA_DEVICE_COMPUTE_ENGINE_COUNT_MIN,        // GNA_UINT32_T
//    GNA_DEVICE_COMPUTE_ENGINE_COUNT_MAX,        // GNA_UINT32_T
//    GNA_DEVICE_ACTIVATION_ENGINE_COUNT,         // GNA_UINT32_T
//    GNA_DEVICE_ACTIVATION_ENGINE_COUNT_MIN,     // GNA_UINT32_T
//    GNA_DEVICE_ACTIVATION_ENGINE_COUNT_MAX,     // GNA_UINT32_T
//    GNA_DEVICE_POOLING_ENGINE_COUNT,            // GNA_UINT32_T
//    GNA_DEVICE_POOLING_ENGINE_COUNT_MIN,        // GNA_UINT32_T
//    GNA_DEVICE_POOLING_ENGINE_COUNT_MAX,        // GNA_UINT32_T
//    GNA_DEVICE_STREAM_COUNT,                    // GNA_UINT32_T
//    GNA_DEVICE_STREAM_COUNT_MIN,                // GNA_UINT32_T
//    GNA_DEVICE_STREAM_COUNT_MAX,                // GNA_UINT32_T
//    GNA_DEVICE_INPUT_BUFFER_SIZE,               // GNA_UINT64_T
//    GNA_DEVICE_MEMORY_MODE,                     // GNA_MEMORY_MODE_T
//    GNA_DEVICE_MEMORY_DEDICATED_SIZE,           // GNA_UINT64_T
//    GNA_DEVICE_MEMORY_REGIONS_COUNT_MAX,        // GNA_UINT32_T
//    GNA_DEVICE_MODEL_SIZE_MAX,                  // GNA_UINT64_T
//    GNA_DEVICE_MODEL_COUNT_MAX,                 // GNA_UINT64_T
//    GNA_DEVICE_MODEL_LAYER_COUNT_MAX,           // GNA_UINT32_T
//    // ANNA
//    GNA_DEVICE_EXT_,           // GNA_UINT32_T
//
//    GNA_DEVICE_PROPERTY_COUNT
//} gna_device_property;
//
//typedef enum _layer_property
//{
//    GNA_LAYER_SUPPORTED,                        // GNA_BOOL_T        // TODO:3:CAPS:Use device's layer support coverage
//    GNA_LAYER_HAS_ACTIVE_LIST,                  // GNA_BOOL_T       // TODO:3:CAPS:Use hw layer's properties
//
//    GNA_LAYER_INPUT_TENSOR_DIM_N_MIN,           // grouping GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_N_MAX,           // grouping GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_W_MIN,           // GNA_UINT32_T         // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_W_MAX,           // GNA_UINT32_T         // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_H_MIN,           // GNA_UINT32_T // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_H_MAX,           // GNA_UINT32_T // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_D_MIN,           // GNA_UINT32_T // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_DIM_D_MAX,           // GNA_UINT32_T // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_INPUT_TENSOR_PRECISION,           // GNA_DATA_MODE_T      // TODO:3:CAPS:Use device's layer support coverage
//    GNA_LAYER_INPUT_TENSOR_ORDER,               // GNA_TENSOR_ORDER_T   // TODO:3:CAPS:Use device's layer support coverage
//
//    GNA_LAYER_OUTPUT_TENSOR_DIM_N_MIN,           // grouping GNA_UINT32_T   // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_N_MAX,           // grouping GNA_UINT32_T   // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_W_MIN,           // GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_W_MAX,           // GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_H_MIN,           // GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_H_MAX,           // GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_D_MIN,           // GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_DIM_D_MAX,           // GNA_UINT32_T    // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_OUTPUT_TENSOR_PRECISION,          // GNA_DATA_MODE_T      // TODO:3:CAPS:Use device's layer support coverage
//    GNA_LAYER_OUTPUT_TENSOR_ORDER,              // GNA_TENSOR_ORDER_T   // TODO:3:CAPS:Use device's layer support coverage
//
//    GNA_LAYER_WEIGHT_TENSOR_PRECISION,          // GNA_DATA_MODE_T      // TODO:3:CAPS:Use device's layer support coverage
//    GNA_LAYER_WEIGHT_TENSOR_ORDER,              // GNA_TENSOR_ORDER_T   // TODO:3:CAPS:Use device's layer support coverage
//
//    GNA_LAYER_BIAS_TENSOR_DIM_N_MIN,            // bias grouping GNA_UINT32_T   // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_BIAS_TENSOR_DIM_N_MAX,            // bias grouping GNA_UINT32_T   // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_BIAS_TENSOR_DIM_W_MIN,            // GNA_UINT32_T // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_BIAS_TENSOR_DIM_W_MAX,            // GNA_UINT32_T // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_BIAS_TENSOR_PRECISION,            // GNA_DATA_MODE_T      // TODO:3:CAPS:Use device's layer support coverage
//    GNA_LAYER_BIAS_TENSOR_ORDER,                // GNA_TENSOR_ORDER_T   // TODO:3:CAPS:Use device's layer support coverage
//
//    GNA_LAYER_ACTIVATION_FUNCTION_MODE,         // GNA_DATA_MODE_T      // TODO:3:CAPS:Use hw layer's properties
//    GNA_LAYER_ACTIVATION_FUNCTION_RELU_HINT,    // GNA_BOOL_T           // TODO:3:CAPS:Use hw layer's properties
//
//    // TODO:3:CAPS:Use hw layer's properties
//    // properties specific for GNA_LAYER_* subject
//    GNA_LAYER_CONVOLUTION_BIAS_VOLUME,          // GNA_BIAS_MODE_T
//    GNA_LAYER_CONVOLUTION_FILTER_COUNT_MIN,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_FILTER_COUNT_MAX,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_FILTER_COUNT_STEP,    // GNA_UINT32_T filter count must be multiple of
//    GNA_LAYER_CONVOLUTION_FILTER_DIM_W_MIN,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_FILTER_DIM_W_MAX,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_FILTER_DIM_H_MIN,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_FILTER_DIM_H_MAX,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_KERNEL_ALIGNMENT,     // GNA_UINT32_T ??? to verify
//    GNA_LAYER_CONVOLUTION_KERNEL_ELEMENT_PRECISION,// GNA_DATA_MODE_T
//    GNA_LAYER_CONVOLUTION_STRIDE_DIM_W_MIN,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_STRIDE_DIM_W_MAX,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_STRIDE_DIM_H_MIN,     // GNA_UINT32_T
//    GNA_LAYER_CONVOLUTION_STRIDE_DIM_H_MAX,     // GNA_UINT32_T
//    //GNA_LAYER_CONVOLUTION_STRIDE_DIM_D_MIN,     // GNA_UINT32_T
//    //GNA_LAYER_CONVOLUTION_STRIDE_DIM_D_MAX,     // GNA_UINT32_T
//
//    GNA_LAYER_POOLING_MODE,                     // GNA_POOLING_MODE_T
//    GNA_LAYER_POOLING_WINDOW_SIZE_DIM_W_MIN,    // GNA_UINT32_T
//    GNA_LAYER_POOLING_WINDOW_SIZE_DIM_W_MAX,    // GNA_UINT32_T
//    GNA_LAYER_POOLING_WINDOW_SIZE_DIM_H_MIN,    // GNA_UINT32_T
//    GNA_LAYER_POOLING_WINDOW_SIZE_DIM_H_MAX,    // GNA_UINT32_T
//    //GNA_LAYER_POOLING_WINDOW_SIZE_DIM_D_MIN,    // GNA_UINT32_T
//    //GNA_LAYER_POOLING_WINDOW_SIZE_DIM_D_MAX,    // GNA_UINT32_T
//    GNA_LAYER_POOLING_STRIDE_DIM_W_MAX,         // GNA_UINT32_T
//    GNA_LAYER_POOLING_STRIDE_DIM_W_MIN,         // GNA_UINT32_T
//    GNA_LAYER_POOLING_STRIDE_DIM_H_MIN,         // GNA_UINT32_T
//    GNA_LAYER_POOLING_STRIDE_DIM_H_MAX,         // GNA_UINT32_T
//    //GNA_LAYER_POOLING_STRIDE_DIM_D_MIN,         // GNA_UINT32_T
//    //GNA_LAYER_POOLING_STRIDE_DIM_D_MAX,         // GNA_UINT32_T
//
//    GNA_LAYER_RECURRENT_FEEDBACK_DEPTH_MIN,     // GNA_UINT32_T
//    GNA_LAYER_RECURRENT_FEEDBACK_DEPTH_MAX,     // GNA_UINT32_T
//} gna_layer_property;
//
///**
// * Test if given mode is set amongst flags
// *
// * @modeFlags   A value or bitwise OR of more values from GNA mode enumeration.
// * @mode        A tested mode value from GNA mode enumeration.
// * @return true if mode is set, false otherwise
//*/
//inline bool GnaIsFlagSet(uint32_t modeFlags, uint32_t mode)
//{
//    if (modeFlags & mode || GNA_NOT_SUPPORTED == mode)
//    {
//        return true;
//    }
//    return false;
//}

// TODO:enable querying some properties on non-available devices like SueScreek

GNAAPI intel_gna_status_t GnaGetDeviceCount(
    uint32_t* deviceCount);
//
//// dedicated query functions
//GNAAPI intel_gna_status_t GnaGetApiProperty(
//    gna_api_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//// optional
//GNAAPI intel_gna_status_t GnaSetApiProperty(
//    gna_api_property property,
//    void* poropertyValue);                      // [in] value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA_LAYER_POOLING_MODE"
//GNAAPI intel_gna_status_t GnaApiPropertyNameToString(
//    gna_api_property property,
//    char const ** propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA_POOLING_MAX | GNA_POOLING_SUM"
//GNAAPI intel_gna_status_t GnaApiPropertyValueToString(
//    gna_api_property property,
//    void* poropertyValue,                       // [in] value of property
//    char const ** propertyString);               // [out] c-string containing property value, allocated by GNA
//
//GNAAPI intel_gna_status_t GnaGetDeviceProperty(
//    gna_device_id device,                       // id/index of device <0;GNA_DEVICE_AVAILABLE_COUNT-1>
//    gna_device_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNAAPI intel_gna_status_t GnaSetDeviceProperty(
//    gna_device_id device,
//    gna_device_property property,
//    void* poropertyValue);                      // [in] value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA_LAYER_POOLING_MODE"
//GNAAPI intel_gna_status_t GnaDevicePropertyNameToString(
//    gna_device_property property,
//    char const * propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA_POOLING_MAX | GNA_POOLING_SUM"
//GNAAPI intel_gna_status_t GnaDevicePropertyValueToString(
//    gna_device_property property,
//    void* poropertyValue,                       // [in] value of property
//    char const * propertyString);               // [out] c-string containing property value, allocated by GNA
//
//GNAAPI intel_gna_status_t GnaGetLayerProperty(
//    gna_device_id device,
//    gna_layer_operation layerOperation,
//    gna_layer_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNAAPI intel_gna_status_t GnaSetLayerProperty(
//    gna_device_id device,
//    gna_layer_operation layerOperation,
//    gna_layer_property property,
//    void* poropertyValue);                      // [in] value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA_LAYER_POOLING_MODE"
//GNAAPI intel_gna_status_t GnaLayerPropertyNameToString(
//    gna_layer_property property,
//    char const * propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA_POOLING_MAX | GNA_POOLING_SUM"
//GNAAPI intel_gna_status_t GnaLayerPropertyValueToString(
//    gna_layer_property property,
//    void* poropertyValue,                       // [in] value of property
//    char const * propertyString);               // [out] c-string containing property value, allocated by GNA
//
//
//// Query hardware device properties even if not present in system, like SueCreek
//GNAAPI intel_gna_status_t GnaGetHardwareDeviceProperty(
//    gna_device_generation generation,         // hardware device generation identifier, for not present devices
//    gna_device_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNAAPI intel_gna_status_t GnaGetHardwareLayerProperty(
//    gna_device_generation generation,         // hardware device generation identifier, for not present devices
//    gna_layer_operation layerOperation,
//    gna_layer_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property


/******************************************************************************
 * GNA Device API
 *****************************************************************************/

///** Maximum number of opened devices */
//const gna_device_id GNA_DEVICE_LIMIT = 1;

/** Device Id indicating invalid device */
const gna_device_id GNA_DEVICE_INVALID = (uint32_t) GNA_DISABLED;

/**
 * Opens and initializes GNA device for processing.
 * NOTE:
 * - The device has to be closed after usage to prevent resource leakage.
 * - Only GNA_DEVICE_ACTIVE_COUNT_MAX number of devices can stay opened at a time.
 *
 * @param threadCount   Number of software worker threads <1,127>. Currently only 1 thread is supported.
 * @param device        (in-out) index/Id of the device to open,  set to GNA_DEVICE_INVALID in case the device can not be opened.
 */
GNAAPI intel_gna_status_t GnaDeviceOpen(
    uint8_t threadCount,
    gna_device_id * device);


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////    end of changes              ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Closes GNA device and releases the corresponding resources.
 *
 * @param device      The device to be closed.
 */
GNAAPI intel_gna_status_t GnaDeviceClose(
    gna_device_id device);

/******************************************************************************
 * GNA Memory API
 *****************************************************************************/

/**
 * Allocates memory buffer, that can be used with GNA device.
 * NOTE:
 * - only 1 allocation at a time is supported
 *
 * @param device      The device which will utilize the allocated buffer.
 * @param sizeRequested Buffer size desired by the caller.
 * @param layerCount    Total number of layers for all neural networks
 * @param gmmCount      Number of gmm layers for all neural networks
 * @param sizeGranted   (out) Buffer size granted by GNA,
 *                      can be less then requested due to HW constraints.
 */
GNAAPI void* GnaAlloc(
    const gna_device_id device,
    const uint32_t sizeRequested,
    const uint16_t layerCount,
    const uint16_t gmmCount,
    uint32_t * sizeGranted);
// TODO:3:add status
// TODO:3:facilitate multiple allocations

/**
 * Releases all allocated memory buffers for given device.
 *
 * @param device      The device which was paired with the buffer.
 */
GNAAPI intel_gna_status_t GnaFree(
    gna_device_id device);

// TODO:3:facilitate multiple allocations
GNAAPI intel_gna_status_t GnaFreeMemory(
    void* memory);

/******************************************************************************
 * GNA Model API
 *****************************************************************************/

/** GNA Model identifier **/
typedef uint32_t gna_model_id;

/** GNA Model type **/
typedef intel_nnet_type_t gna_model;

/**
 * Creates and compiles the model for use with a given device.
 * NOTE:
 * - Only 1 model supported in the first phase.
 * - All Model's data has to be placed in single memory buffer allocated previously by GNAAlloc.
 * - The descriptor has to be placed in user's memory, not allocated by GNAAlloc.
 *
 * @param device      GNA device that will utilize the model.
 * @param model         Model descriptor which will govern the model creation.
 * @param modelId       (out) The model created by GNA.
 */
GNAAPI intel_gna_status_t GnaModelCreate(
    gna_device_id device,
    gna_model const * model,
    gna_model_id * modelId);

/******************************************************************************
 * GNA Inference API
 *****************************************************************************/

/** GNA Request configuration identifier **/
typedef uint32_t gna_request_cfg_id;

/**
 * Component type
 * Used e.g. for RequestConfig buffers.
 **/
typedef enum _ComponentType
{
    InputComponent = 0,
    OutputComponent = 1,
    IntermediateOutputComponent,
    WeightComponent,
    FilterComponent = WeightComponent,
    BiasComponent,
    WeightScaleFactorComponent,
    PwlComponent,
    StrideComponent,
    WindowComponent,
    // TODO:3: CopyComponent?,
    GmmMeanComponent,
    GmmInverseCovarianceComponent,
    GmmGaussianConstantComponent,
    // TODO:3: Recurrent component?
    ComponentTypeCount,
} GnaComponentType;

/**
 * Adds single request configuration for use with the model.
 * Request configurations have to be declared a priori to minimize the
 * request preparation time and reduce processing latency.
 * This configuration holds buffers that can be used with consecutive requests
 * to handle asynchronous processing.
 * When requests are processed asynchronously each one needs to have individual
 * Input and output buffers set by this configuration.
 * Configurations can be reused with another request when the request
 * with the current configuration has been completed and retrieved by GnaRequestWait.
 * Eg. The user can create 8 unique configurations and reuse them
 * with consecutive batches of 8 requests, when batches are enqueued sequentially.
 * NOTE:
 * - Unreleased configurations are released by GNA during corresponding model release.
 *
 * @param modelId       The model that utilizes the request configuration.
 *                      Request configuration cannot be shared with other models.
 * @param configId      (out) Request configuration created by GNA.
 */
GNAAPI intel_gna_status_t GnaModelRequestConfigAdd(
    gna_model_id modelId,
    gna_request_cfg_id * configId);

/**
 * Adds a single buffer to the request configuration.
 * Subsequent calls add consecutive buffers to the list.
 * Each request configuration needs to have at least
 * - 1 input buffer for the first layer
 * - and 1 output buffer for the last layer.
 *
 * @see GnaRequestConfigActiveListAdd Can be used to add Active list to the model's output.
 *
 * NOTE:
 * - Buffer addresses need to be within the memory allocated previously by GNAAlloc.
 * - Buffers are deleted by GNA during corresponding request configuration release.
 *
 * @param configId      Request configuration to pair with the buffer.
 * @param type          Type of the buffer being added.
 * @param layerIndex    Index of the layer that hosts the buffer.
 * @param address       Address of the buffer.
 */
GNAAPI intel_gna_status_t GnaRequestConfigBufferAdd(
    gna_request_cfg_id configId,
    GnaComponentType type,
    uint32_t layerIndex,
    void * address);

/**
 * Adds active outputs list to the request configuration.
 * Active output list can be specified only for the output (usually last) layer(s).
 * Only one active list can be added to each output layer.
 * To modify indices for consecutive requests during sequential processing
 * user can modify the content of the indices buffer.
 * To modify indices for consecutive requests during parallel processing
 * user can create additional configurations with appropriate parameters.
 * NOTE:
 * - Active lists are deleted by GNA during corresponding request configuration release.
 * - Buffer addresses need to be within the memory allocated previously by GNAAlloc.
 *
 * @param configId      Request configuration which will utilize the active list.
 * @param layerIndex    Index of the layer that active list is specified for.
 *                      The layer needs to have a OutputComponent type buffer
 *                      already assigned in the request configuration.
 * @param indicesCount  The number of indices in the active list.
 * @param indices       The address of the array with active output indices.
 * @see GnaModelRequestConfigAdd and GnaRequestConfigBufferAdd for details.
 */
GNAAPI intel_gna_status_t GnaRequestConfigActiveListAdd(
    gna_request_cfg_id configId,
    uint32_t layerIndex,
    uint32_t indicesCount,
    uint32_t const * indices);

/**
 * The list of processing acceleration modes.
 * GNA supports a bunch of acceleration modes. Their availability depends on the CPU type.
 * The modes supported by the current system are detected by GNA.
 * Use GNA_AUTO mode to let GNA select the best available acceleration.
 * NOTE:
 * - GNA_HARDWARE: in some GNA hardware generations, model components unsupported
 *   by hardware will be processed using software acceleration.
 * By default fast acceleration mode, which does not detect saturation will be used.
 * @see GnaSetSaturationDetection to enable saturation detection.
 */
typedef enum  _acceleration
{
    GNA_HARDWARE = 0xFFFFFFFE,   // GNA Hardware acceleration enforcement
    GNA_AUTO     = 0x3,          // GNA selects the best available acceleration
    GNA_SOFTWARE = 0x5,          // GNA selects the best available software acceleration
    GNA_GENERIC  = 0x7,          // Enforce the usage of generic software mode
    GNA_SSE4_2   = 0x9,          // Enforce the usage of SSE 4.2 CPU instruction set
    GNA_AVX1     = 0xB,          // Enforce the usage of AVX1 CPU instruction set
    GNA_AVX2     = 0xD           // Enforce the usage of AVX2 CPU instruction set
} gna_acceleration;

static_assert(4 == sizeof(gna_acceleration), "Invalid size of gna_acceleration");

/**
 * Sets saturation detection for a given acceleration mode.
 * Use only for GnaRequestEnqueue acceleration argument.
 * Hardware acceleration has saturation detection always enabled.
 * GMM layers have saturation detection always enabled.
 *
 * @param acceleration  The desired acceleration mode.
 * @return Acceleration mode with enabled saturation detection.
 */
inline gna_acceleration GnaSetSaturationDetection(
    const gna_acceleration acceleration)
{
    return (gna_acceleration)(acceleration & GNA_HARDWARE);
}

/** GNA Request identifier **/
typedef uint32_t gna_request_id;

/** GNA Wait Timeout type **/
typedef uint32_t gna_timeout;

/**
 * Creates and enqueues a request for asynchronous processing.
 * NOTE:
 * - Request's life cycle and memory is managed by GNA.
 * - The model, that the request will be calculated against is provided by configuration.
 *
 * @param configId      The request configuration.
 * @param acceleration  Acceleration mode used for processing.
 * @param requestId     (out) Request created by GNA.
 * @return              Status of request preparation and queuing only.
 *                      To retrieve the results and processing status call GnaRequestWait.
 */
GNAAPI intel_gna_status_t GnaRequestEnqueue(
    gna_request_cfg_id configId,
    gna_acceleration acceleration,
    gna_request_id * requestId);

/**
 * Waits for the request processing to be completed.
 * NOTE:
 * - If processing is completed before the timeout expires, the request object is released.
 *   Otherwise processing status is returned.
 * - Unretrieved requests are released by GNA during corresponding model release.
 *
 * @param requestId     The request to wait for.
 * @param milliseconds  timeout duration in milliseconds.
 */
GNAAPI intel_gna_status_t GnaRequestWait(
    gna_request_id requestId,
    gna_timeout milliseconds);

/** Maximum number of requests that can be enqueued before retrieval */
const uint32_t GNA_REQUEST_QUEUE_LENGTH = 64;

/** Request Id indicating that GnaRequestWait should wait until any request completes. */
const gna_request_id GNA_REQUEST_WAIT_ANY = 0xffffffff;

/** Maximum supported time of waiting for a request */
const gna_timeout GNA_REQUEST_TIMEOUT_MAX = 180000;


/******************************************************************************
 * GNA Utilities API
 *****************************************************************************/

/**
 * Rounds a number up, to the nearest multiple of significance
 * Used for calculating the memory sizes of GNA data buffers
 *
 * @param number        Memory size or a number to round up.
 * @param significance  Informs the function how to round up. The function "ceils"
 *                      the number to the lowest possible value divisible by "significance".
 * @return Rounded integer value.
 * @deprecated          Will be removed in next release.
 */
#define ALIGN(number, significance)   (((unsigned int)((number) + significance -1) / significance) * significance)

/**
 * Rounds a number up, to the nearest multiple of 64
 * Used for calculating memory sizes of GNA data arrays
 * @deprecated          Will be removed in next release.
 */
#define ALIGN64(number)   ALIGN(number, 64)

/**
 * Verifies data sizes used in the API and GNA hardware
 *
 * NOTE: If data sizes in an application using API differ from data sizes
 *       in the API library implementation, scoring will not work properly
 */
static_assert(1 == sizeof(int8_t), "Invalid size of int8_t");
static_assert(2 == sizeof(int16_t), "Invalid size of int16_t");
static_assert(4 == sizeof(int32_t), "Invalid size of int32_t");
static_assert(1 == sizeof(uint8_t), "Invalid size of uint8_t");
static_assert(2 == sizeof(uint16_t), "Invalid size of uint16_t");
static_assert(4 == sizeof(uint32_t), "Invalid size of uint32_t");

#ifdef __cplusplus
}
#endif

#endif  // ifndef __GNA_API_H
