/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

/******************************************************************************
 *
 * GNA 2.0 API
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * API Definition
 *
 *****************************************************************************/

#ifndef __GNA_API_H
#define __GNA_API_H

#include <stdint.h>

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Library API import/export macros */
#if 1 == _WIN32
#   if 1 == INTEL_GNA_DLLEXPORT
#       define GNAAPI __declspec(dllexport)
#   else
#       define GNAAPI __declspec(dllimport)
#   endif
#else
#       define GNAAPI
#endif


/******************  GNA Device API ******************/

/** GNA Device identifier **/
typedef uint32_t gna_device_id;

/** Maximum number of opened devices */
const gna_device_id GNA_DEVICE_LIMIT = 1;

/** Device Id indicating invalid device */
const gna_device_id GNA_DEVICE_INVALID = 0;

/**
 * Opens and initializes GNA device for processing.
 * NOTE:
 * - The device has to be closed after usage to prevent resource leakage.
 * - Only GNA_DEVICE_LIMIT number of devices can stay opened at a time.
 *
 * @param threadCount   Number of software worker threads <1,127>. Currently only 1 thread is supported.
 * @param deviceId      (out) Id of the device that got opened or GNA_DEVICE_INVALID in case the device can not be opened.
 */
GNAAPI intel_gna_status_t GnaDeviceOpen(
    uint8_t threadCount,
    gna_device_id * deviceId);

/**
 * Closes GNA device and releases the corresponding resources.
 *
 * @param deviceId      The device to be closed.
 */
GNAAPI intel_gna_status_t GnaDeviceClose(
    gna_device_id deviceId);

/******************  GNA Memory API ******************/
/***** @deprecated Will be removed in next release. **/

/**
 * Allocates memory buffer, that can be used with GNA device.
 * NOTE:
 * - only 1 allocation at a time is supported
 *
 * @param deviceId      The device which will utilize the allocated buffer.
 * @param sizeRequested Buffer size desired by the caller.
 * @param layerCount    Total number of layers for all neural networks
 * @param gmmCount      Number of gmm layers for all neural networks
 * @param sizeGranted   (out) Buffer size granted by GNA,
 *                      can be less then requested due to HW constraints.
 * @deprecated          Will be removed in next release.
 */
GNAAPI void* GnaAlloc(
    const gna_device_id deviceId,
    const uint32_t sizeRequested,
    const uint16_t layerCount,
    const uint16_t gmmCount,
    uint32_t * sizeGranted);

/**
 * Releases the memory buffer.
 *
 * @param deviceId      The device which was paired with the buffer.
 * @deprecated          Will be removed in next release.
 */
GNAAPI intel_gna_status_t GnaFree(
    gna_device_id deviceId);


/******************  GNA Model API ******************/

/** GNA Model identifier **/
typedef uint32_t gna_model_id;

/** GNA Model type **/
typedef intel_nnet_type_t gna_model;

/**
 * Creates and compiles the model for use with a given device.
 * NOTE:
 * - Only 1 model supported in the first phase.
 * - Model's data has to be placed in memory allocated previously by GNAAlloc.
 * - The descriptor has to be placed in user's memory, not allocated by GNAAlloc.
 *
 * @param deviceId      GNA device that will utilize the model.
 * @param model         Model descriptor which will govern the model creation.
 * @param modelId       (out) The model created by GNA.
 */
GNAAPI intel_gna_status_t GnaModelCreate(
    gna_device_id deviceId,
    gna_model const * model,
    gna_model_id * modelId);

/******************  GNA Request Configuration API ******************/

/** GNA Request configuration identifier **/
typedef uint32_t gna_request_cfg_id;

/** Buffer type for request configuration. */
typedef enum _buffer_type {
    GNA_IN,             // Input buffer read by GNA device
    GNA_OUT,            // Output buffer that GNA will write to
    GNA_BUFFER_TYPES
} gna_buffer_type;

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
    gna_buffer_type type,
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
 *                      The layer needs to have a GNA_OUT type buffer
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


/******************  GNA Request Calculation API ******************/

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


/******************  GNA Utilities API ******************/

/**
 * Gets printable status name with the description as a c-string
 *
 * @param status        A status to translate.
 * @return A c-string status with the description.
 */
GNAAPI char const * GnaStatusToString(
    intel_gna_status_t status);

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
