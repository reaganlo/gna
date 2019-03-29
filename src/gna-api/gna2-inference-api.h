/*
 @copyright

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-inference-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA_API_INFERENCE Inference API

 API for configuring and running inference requests.

 @{
 *****************************************************************************/

#ifndef __GNA2_INFERENCE_API_H
#define __GNA2_INFERENCE_API_H

#include "gna2-common-api.h"

#include <stdint.h>

enum GnaAccelerationMode;

/**
 Adds single request configuration for use with the model.

 Request configurations have to be declared a priori to minimize the
 request preparation time and reduce processing latency.
 This configuration holds buffers that can be used with consecutive requests
 to handle asynchronous processing.
 When requests are processed asynchronously each one needs to have individual
 Input and output buffers set by this configuration.
 Configurations can be reused with another request when the request
 with the current configuration has been completed and retrieved by GnaRequestWait().
 Eg. The user can create 8 unique configurations and reuse them
 with consecutive batches of 8 requests, when batches are enqueued sequentially.

 @note
 - Unreleased configurations are released by GNA during corresponding model release.

 @param modelId The model that utilizes the request configuration.
                Request configuration cannot be shared with other models.
 @param [out] requestConfigId Request configuration created by GNA.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaRequestConfigCreate(
    uint32_t modelId,
    uint32_t * requestConfigId);

/**
 Sets a buffer of the operation operand for the request configuration.

 @note
 - Buffer addresses need to be within the memory allocated previously by GnaMemoryAlloc.
 - Buffers are deleted by GNA during corresponding request configuration release. // TODO:3: is this true?

 @param requestConfigId Identifier of affected request configuration.
 @param operationIndex Index of the affected operation.
 @param operandIndex Index of the affected operand.
 @param address Address of the buffer.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaRequestConfigSetOperandBuffer(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t operandIndex,
    void * address);

/**
 Adds active outputs list to the request configuration.

 Active output list can be specified only for the output (usually last) operation(s).
 Only one active list can be added to each output operation.
 To modify indices for consecutive requests during sequential processing
 user can modify the content of the indices buffer.
 To modify indices for consecutive requests during parallel processing
 user can create additional configurations with appropriate parameters.

 @note
 - Active lists are deleted by GNA during corresponding request configuration release.
 - Buffer addresses need to be within the memory allocated previously by GNAAlloc.

 @param requestConfigId Identifier of affected request configuration.
 @param operationIndex Index of the affected operation.
                       The operation must have an output buffer already assigned.
 @param numberOfIndices The number of indices in the active list.
 @param indices The address of the array with active output indices.
 @return Status of the operation.
 @see GnaRequestConfigCreate and GnaRequestConfigSetOperandBuffer for details.
 */
GNA_API enum GnaStatus GnaRequestConfigEnableActiveList(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t numberOfIndices,
    uint32_t const * indices);

/**
 Enables software result consistency with selected device version.

 Assures that for given request config software mode inference results
 (scores) are bit-exact with that produced by the hardware device .
 Useful e.g. for verification of results from model created and exported
 for GNA embedded devices.
 @see GnaAccelerationMode.

 @param requestConfigId Identifier of affected request configuration.
 @param deviceVersion Device version to be consistent with.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaRequestConfigEnableHardwareConsistency(
    uint32_t requestConfigId,
    enum GnaDeviceVersion deviceVersion);

/**
 Enforces processing request with selected acceleration mode.

 When not set ::GnaAccelerationModeAuto is used.
 @see GnaAccelerationMode.

 @param requestConfigId Identifier of affected request configuration.
 @param accelerationMode Acceleration mode used for processing.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaRequestConfigSetAccelerationMode(
    uint32_t requestConfigId,
    enum GnaAccelerationMode accelerationMode);

/**
 The list of processing acceleration modes.

 Current acceleration modes availability depends on the CPU type.
 Available modes are detected by GNA.

 @note
 - ::GnaAccelerationModeHardware: in some GNA hardware generations, model components unsupported
   by hardware will be processed using software acceleration.

 When software inference is used, by default "fast" algorithm is used
 and results may be not bit-exact with these produced by hardware device.
 @see GnaRequestConfigEnableHardwareConsistency to enable bit-exact results consistency.
 */
enum GnaAccelerationMode
{
    /**
     Fully automated acceleration selection.

     GNA library, based on availability, selects hardware mode or the best
     (highest performance) applicable software acceleration.

     The order of preference is as follows:
         1. ::GnaAccelerationModeHardware,
         2. ::GnaAccelerationModeAvx2,
         3. ::GnaAccelerationModeAvx1,
         4. ::GnaAccelerationModeSse4x2
         5. ::GnaAccelerationModeGeneric.
     */
    GnaAccelerationModeAuto = GNA_DEFAULT,

    /**
     Automated software acceleration selection.

     Automatic software emulation selection similar as in ::GnaAccelerationModeAuto.
     Only software optimizations are selected, even if GNA hardware device is available.
     */
    GnaAccelerationModeSoftware = 1,

    /**
     GNA Hardware device acceleration.

     Enforces the usage of GNA Hardware acceleration.
     Hardware acceleration has saturation detection always enabled.
     For some older GNA hardware generations, model components unsupported
     by hardware will be processed using software acceleration.
     */
    GnaAccelerationModeHardware = 2,

    /**
     AVX2 Software acceleration.

     Enforce the usage of optimized software implementation,
     using AVX2 CPU instruction set.
     */
    GnaAccelerationModeAvx2 = 3,

    /**
     AVX1 Software acceleration.

     Enforce the usage of optimized software implementation,
     using AVX1 CPU instruction set.
     */
    GnaAccelerationModeAvx1 = 4,

    /**
     SSE4.2 Software acceleration.

     Enforce the usage of optimized software implementation,
     using SSE4.2 CPU instruction set.
     */
    GnaAccelerationModeSse4x2 = 5,

    /**
     Generic software implementation.

     Enforce the usage of generic software implementation,
     using basic x86 CPU instruction set.
     */
    GnaAccelerationModeGeneric = 6,
};

/**
 Releases request config and its resources.

 @note
 - Not thread-safe.
 - Please make sure all requests using this config are completed.

 @param requestConfigId Identifier of affected request configuration.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaRequestConfigRelease(
    uint32_t requestConfigId);

/**
 Creates and enqueues a request for asynchronous processing.

 @note
 - Request's life cycle and memory is managed by GNA.
 - The model, that the request will be calculated against is provided by configuration.
 - Maximum number of requests that can wait in the queue at once is 64.

 @param requestConfigId The request configuration.
 @param [out] requestId Identifier of the enqueued request.
 @return Status of request preparation and queuing only. To retrieve
         the results and processing status call GnaRequestWait.
 */
GNA_API enum GnaStatus GnaRequestEnqueue(
    uint32_t requestConfigId,
    uint32_t * requestId);

/**
 Waits for the request processing to be completed.

 @note
 - If processing is completed before the timeout expires, the request object is released.
   Otherwise processing status is returned.
 - Unretrieved requests are released by GNA during corresponding model release.
 - Maximum supported time of waiting for a request is 180000 milliseconds.

 @param requestId The request to wait for, use ::GNA_DISABLED to wait for any.
 @param timeoutMilliseconds Timeout duration in milliseconds.
 @return Status of request processing.
 */
GNA_API enum GnaStatus GnaRequestWait(
    uint32_t requestId,
    uint32_t timeoutMilliseconds);

#endif // __GNA2_INFERENCE_API_H

/**
 @}
 @}
 */
