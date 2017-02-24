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

/** GNA Device identification **/
typedef uint32_t gna_device_id;


/** Maximum number of opened devices */
const gna_device_id GNA_DEVICE_LIMIT = 1;

/** Device Id indicating invalid device */
const gna_device_id GNA_DEVICE_INVALID = 0;

/**
 * Opens and initializes GNA device for processing.
 * NOTE:
 * - Device have to be closed after user finished using GNA
 *   to prevent resource leakage.
 * - Only GNA_DEVICE_LIMIT number of devices can stay opened at one time.
 *
 * @param threadCount   Number of software worker threads <1,127>. Currently only 1 thread is supported.
 * @param deviceId      (out) Device opened by GNA.
 * @return New device Id or GNA_DEVICE_INVALID in case device cannot be opened.
 */
GNAAPI intel_gna_status_t GnaDeviceOpen(
    uint8_t             threadCount,
    gna_device_id*      deviceId);

/**
 * Closes GNA device and releases corresponding resources.
 *
 * @param deviceId      Device to be closed.
 */
GNAAPI intel_gna_status_t GnaDeviceClose(
    gna_device_id       deviceId);

/******************  GNA Memory API ******************/
/***** @deprecated Will be removed in next release. **/

/**
 * Allocates memory buffer, that can be used with GNA device.
 * NOTE:
 * - only 1 allocation at time is supported
 *
 * @param deviceId      Device, that buffer will be used with.
 * @param sizeRequested Buffer size desired by caller.
 * @param sizeGranted   (out) Buffer size granted by GNA,
 *                      size can differ to meet device requirements.
 * @deprecated          Will be removed in next release.
 */
GNAAPI void* GnaAlloc(
    gna_device_id       deviceId,
    uint32_t            sizeRequested,
    uint32_t*           sizeGranted);

/**
 * Releases memory buffer.
 *
 * @param deviceId      Device, that buffer was assigned to.
 * @deprecated          Will be removed in next release.
 */
GNAAPI intel_gna_status_t GnaFree(
    gna_device_id       deviceId);


/******************  GNA Model API ******************/

/** GNA Model identification **/
typedef uint32_t gna_model_id;

/** GNA Model type **/
typedef intel_nnet_type_t gna_model;

/**
 * Creates and compiles model for use with given device.
 * NOTE:
 * - Only 1 model supported in first phase.
 * - Model data have to be placed in memory allocated previously by GNAAlloc.
 * - Descriptor have to be placed user memory, not allocated by GNAAlloc.
 *
 * @param deviceId      GNA device, that model will be used with.
 * @param model         Model descriptor that which model will be created from.
 * @param modelId       (out) Model created by GNA.
 */
GNAAPI intel_gna_status_t GnaModelCreate(
    gna_device_id       deviceId,
    gna_model*          model,
    gna_model_id*       modelId);

/**
 * Removes model and releases resources.
 * NOTE:
 * - In first phase memory for model data have to be released by GNAFree.
 * - Unreleased models are released by GNA with device close.
 *
 * @param modelId       Model to be released.
 */
GNAAPI intel_gna_status_t GnaModelRelease(
    gna_model_id        modelId);


/******************  GNA Request Configuration API ******************/

/** GNA Request configuration identification **/
typedef uint32_t gna_request_cfg_id;

/** Type of buffer for request configuration. */
typedef enum _buffer_type {
    GNA_IN,             // Input buffer read by GNA device
    GNA_OUT,            // Output buffer GNA will write to

    GNA_BUFFER_TYPES

} gna_buffer_type;

/**
 * Adds single request configuration for use with model.
 * Request configurations have to be declared a priori to minimize the time
 * of preparation of request and reduce processing latency.
 * This configuration holds buffers that can be used with consecutive requests
 * to handle asynchronous processing.
 * When request are processed asynchronously each need to have individual
 * Input and output buffers set by this configuration.
 * Configurations can be reused with another request when request
 * with current configuration has been completed and retrieved by GnaRequestWait.
 * I.e. User can create e.g. 8 unique configurations and reuse them
 * with consecutive batches of 8 requests, when batches are enqueued sequentially.
 * NOTE:
 * - Unreleased configurations are released by GNA with corresponding model release.
 *
 * @param modelId       Model, that request configuration will be used with.
 *                      Configuration cannot be shared with other models.
 * @param configId      (out) Request configuration created by GNA.
 */
GNAAPI intel_gna_status_t GnaModelRequestConfigAdd(
    gna_model_id        modelId,
    gna_request_cfg_id* configId);

/**
 * Adds single buffer to request configuration.
 * Subsequent calls add consecutive buffers to list.
 * Each request configuration have to have at least
 * - 1 input buffer for first layer
 * - and 1 output buffer for last layer.
 * More buffers can be added to provide additional customization for requests.
 *
 * @see GnaRequestConfigActiveListAdd Can be used to add Active list to model output.
 *
 * NOTE:
 * - Buffer addresses have to be within memory allocated previously by GNAAlloc.
 * - Buffers are deleted by GNA with corresponding request configuration release.
 *
 * @param configId      Request configuration, that buffer will be added to.
 * @param type          Type of buffer being added.
 * @param layerIndex    Index of layer that buffer is specified for.
 * @param address       Address of buffer, that will be used by request.
 */
GNAAPI intel_gna_status_t GnaRequestConfigBufferAdd(
    gna_request_cfg_id  configId,
    gna_buffer_type     type,
    uint32_t            layerIndex,
    void*               address);

/**
 * Adds active outputs list to request configuration.
 * Active output list can be specified only for output (usually last) layer(s).
 * Only one active list can be added to each output layer.
 * To modify indices for consecutive requests during sequential processing
 * user can modify indices buffer contents.
 * To modify indices for consecutive requests during parallel processing
 * user can create additional configurations with appropriate parameters.
 * NOTE:
 * - Active lists are deleted by GNA with corresponding configuration release.
 * - Buffer addresses have to be within memory allocated previously by GNAAlloc.
 *
 * @param configId      Request configuration, that active list will be added to.
 * @param layerIndex    Index of layer that active list is specified for.
 *                      Layer has to have buffer of type GNA_OUT
 *                      already assigned in request configuration.
 * @param indicesCount  Number of active list indices.
 * @param indices       Address of array with active output indices.
 * @see GnaModelRequestConfigAdd and GnaRequestConfigBufferAdd for details.
 */
GNAAPI intel_gna_status_t GnaRequestConfigActiveListAdd(
    gna_request_cfg_id  configId,
    uint32_t            layerIndex,
    uint32_t            indicesCount,
    uint32_t*           indices);


/******************  GNA Request Calculation API ******************/

/**
 * List of processing acceleration modes.
 * GNA supports a bunch of acceleration modes which availability depends on CPU type.
 * Modes supported by current system are detected by GNA.
 * Use GNA_AUTO mode to let GNA select best available acceleration.
 * NOTE:
 * - GNA_HARDWARE: in some GNA hardware generations model components unsupported
 *   by hardware will be processed using software acceleration.
 * By default fast acceleration mode, which does not detect saturation will be used.
 * @see GnaSetSaturationDetection to enable saturation detection.
 */
typedef enum  _acceleration
{
    GNA_HARDWARE = 0xFFFFFFFE,   // GNA Hardware acceleration enforcement
    GNA_AUTO     = 0x3,          // GNA selects best available acceleration
    GNA_SOFTWARE = 0x5,          // GNA selects best available software acceleration
    GNA_GENERIC  = 0x7,          // Enforce use of generic software mode
    GNA_SSE4_2   = 0x9,          // Enforce use of SSE 4.2 CPU instruction set
    GNA_AVX1     = 0xB,          // Enforce use of AVX1 CPU instruction set
    GNA_AVX2     = 0xD           // Enforce use of AVX2 CPU instruction set
} gna_acceleration;

static_assert(4 == sizeof(gna_acceleration), "Invalid size of gna_acceleration");

/**
 * Sets saturation detection for given acceleration mode.
 * Use only for GnaRequestEnqueue acceleration argument.
 * Hardware acceleration has saturation detection always enabled.
 * GMM layers have saturation detection always enabled.
 *
 * @param acceleration  Desired acceleration mode.
 * @return Acceleration mode with enabled saturation detection.
 */
inline gna_acceleration GnaSetSaturationDetection(
    gna_acceleration    acceleration)
{
    return (gna_acceleration)(acceleration & GNA_HARDWARE);
}

/** GNA Request identification **/
typedef uint32_t gna_request_id;

/** GNA Wait Timeout type **/
typedef uint32_t gna_timeout;

/**
 * Creates and enqueues request for asynchronous processing.
 * NOTE:
 * - Request life cycle and memory is managed by GNA.
 * - Model, that request will be calculated against is provided by configuration.
 *
 * @param configId      Request configuration.
 * @param acceleration  Acceleration mode used for processing.
 * @param requestId     (out) Request created by GNA.
 * @return              Status of request preparation and queuing only.
 *                      To retrieve results and processing status call GnaRequestWait.
 */
GNAAPI intel_gna_status_t GnaRequestEnqueue(
    gna_request_cfg_id  configId,
    gna_acceleration    acceleration,
    gna_request_id*     requestId);

/**
 * Waits for request processing to be completed.
 * NOTE:
 * - If processing is completed before timeout, request object is released.
 *   Otherwise processing status is returned.
 * - Unretrieved request are released by GNA with corresponding model release.
 *
 * @param requestId     Request to wait for.
 * @param milliseconds  timeout duration in milliseconds.
 */
GNAAPI intel_gna_status_t GnaRequestWait(
    gna_request_id      requestId,
    gna_timeout         milliseconds);

/** Maximum number of requests that can be enqueued before retrieval */
const uint32_t GNA_REQUEST_QUEUE_LENGTH = 64;

/** Request Id indicating that GnaRequestWait should wait until any request completes. */
const gna_request_id GNA_REQUEST_WAIT_ANY = 0xffffffff;

/** Maximum supported time of waiting for request */
const gna_timeout GNA_REQUEST_TIMEOUT_MAX = 180000;


/******************  GNA Utilities API ******************/

/**
 * Gets printable status name with description as c-string
 *
 * @param status        Status name to retrieve.
 * @return C-string status with description.
 */
GNAAPI const char* GnaStatusToString(
    intel_gna_status_t        status);

/**
 * Rounds a number up, to the nearest multiple of significance
 * Used for calculating memory sizes of GNA data buffers
 *
 * @param number        Memory size or number to round up.
 * @param significance  The multiple to which number will be rounded.
 * @return Rounded integer value.
 * @deprecated          Will be removed in next release.
 */
#define ALIGN(number, significance)   (((int)((number) + significance -1) / significance) * significance)

/**
 * Rounds a number up, to the nearest multiple of 64
 * Used for calculating memory sizes of GNA data arrays
 * @deprecated          Will be removed in next release.
 */
#define ALIGN64(number)   ALIGN(number, 64)

/**
 * Verifies data sizes used in API and hardware
 *
 * NOTE: If data sizes in application using API differ from data sizes
 *       in API library implementation scoring will not work properly
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
