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
 @file gna2-common-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_COMMON_API Common API

 API with commonly used and auxiliary declarations.

 @{
 *****************************************************************************/

#ifndef __GNA2_COMMON_API_H
#define __GNA2_COMMON_API_H

#include <stdint.h>

/**
 C-linkage macro.
 */
#ifdef __cplusplus
#define GNA_API_C extern "C"
#else
#define GNA_API_C
#endif

/**
 Library API import/export macro.
 */
#if !defined(GNA_API_EXPORT)
#    if 1 == _WIN32
#       if 1 == INTEL_GNA_DLLEXPORT
#           define GNA_API_EXPORT __declspec(dllexport)
#       else
#           define GNA_API_EXPORT __declspec(dllimport)
#       endif
#    else
#        if __GNUC__ >= 4
#           define GNA_API_EXPORT __attribute__ ((visibility ("default")))
#        else
#           define GNA_API_EXPORT
#        endif
#    endif
#endif

/**
 Library C-API import/export macro.
 */
#define GNA_API GNA_API_C GNA_API_EXPORT

/** Constant indicating that feature is disabled. */
#define GNA_DISABLED (-1)

/** Constant indicating that value is default. */
#define GNA_DEFAULT (0)

/** Constant indicating that feature is not available. */
#define GNA_NOT_SUPPORTED (1 << 31)

/**
 List of device versions.

 Version determines concrete GNA device derivative.
 Devices of the same version are always single generation and have the same
 properties e.g. frequency, bandwidth.

 @see GnaDeviceGeneration
 */
enum GnaDeviceVersion
{
    /**
     Gaussian Mixture Models device.
     Including Skyake and derivatives, e.g. KabyLake, CoffeLake,
     with the same feature set.
     A ::GnaDeviceGenerationGmm generation device.
     */
    GnaDeviceVersionSkylake = 0x1911,

    /**
     GNA CannonLake (CNL) device.
     A ::GnaDeviceGeneration0x9 generation device.
     */
    GnaDeviceVersionCannonlake = 0x5A11,

    /**
     GNA GeminiLake (GLK) device.
     A ::GnaDeviceGeneration1x0 generation device.
     */
    GnaDeviceVersionGeminilake = 0x3190,

    /**
     GNA ElkhartLake (EHL) device.
     A ::GnaDeviceGeneration1x0 generation device.
     */
    GnaDeviceVersionElkhartLake = 0x4511,

    /**
     GNA IceLake (ICL) device.
     A ::GnaDeviceGeneration1x0 generation device.
     */
    GnaDeviceVersionIcelake = 0x8A11,

    /**
     GNA TigerLake (TGL) device.
     A ::GnaDeviceGeneration2x0 generation device.
     */
    GnaDeviceVersionTigerlake = 0x9A11,

    /**
     GNA AlderLake (ADL) device.
     A ::GnaDeviceGeneration3x0 generation device.
     */
    GnaDeviceVersionAlderLake = 0x46AD,

    /**
     GNA SueCreek (SUE) embedded device.
     A ::GnaDeviceGenerationEmbedded1x0 generation device.
     */
    GnaDeviceVersionSueCreek = 0xFFFF0001,

    /**
     GNA JellyFish (JLF) embedded device.
     A ::GnaDeviceGeneration2x0 generation device.
     */
    GnaDeviceVersionJellyfish = 0xFFFF0002,

    /**
     GNA AlderLake (JLF) embedded device on PCH/ACE.
     A ::GnaDeviceGenerationEmbedded3x0 generation device.
     */
    GnaDeviceVersionAceEmbedded = 0xFFFF0003,

    /**
     GNA ANNA autonomous embedded device on Alder Lake PCH/ACE.
     A ::GnaDeviceGenerationAutonomus3x1 generation device.
     */
    GnaDeviceVersionAceAnna = 0xFFFF0004,

    /**
     Value indicating no supported hardware device available.
     Software emulation (fall-back) will be used.

     @see ::GNA_DEFAULT_DEVICE_VERSION and GnaRequestConfigEnableHardwareConsistency().
     */
    GnaDeviceVersionSoftwareEmulation = GNA_DEFAULT,
};

/**
 Version of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @see
 GnaRequestConfigEnableHardwareConsistency() to change hardware device
 version in software mode.

 @note
 Usually it will be the latest existing GNA device (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA_DEFAULT_DEVICE_VERSION GnaDeviceVersionAlderLake

/**
 GNA API Status codes.
 */
enum GnaStatus
{
    /**
     Success: Operation completed successfully without errors or warnings.
     */
    GnaStatusSuccess = GNA_DEFAULT,

    /**
     Warning: Arithmetic saturation.
     An arithmetic operation has resulted in saturation during calculation.
     */
    GnaStatusWarningArithmeticSaturation = 1,

    /**
     Warning: Device is busy.
     GNA is still running, can not enqueue more requests.
     */
    GnaStatusWarningDeviceBusy = 2,

    /**
     Error: Unknown error occurred.
     */
    GnaStatusUnknownError = -3,

    /**
     Error: Functionality not implemented yet.
     */
    GnaStatusNotImplemented = -4,

    /**
     Error: Item identifier is invalid.
     Provided item (e.g. device or request) id or index is invalid.
    */
    GnaStatusIdentifierInvalid = -5,

     /**
     Error: NULL argument is not allowed.
    */
    GnaStatusNullArgumentNotAllowed = -6,

    /**
     Error: NULL argument is required.
    */
    GnaStatusNullArgumentRequired = -7,

    /**
     Error: Unable to create new resources.
    */
    GnaStatusResourceAllocatonError = -8,

    /**
     Error: Device: not available.
    */
    GnaStatusDeviceNotAvailable = -9,

    /**
     Error: Device failed to open, thread count is invalid.
    */
    GnaStatusDeviceNumberOfThreadsInvalid = -10,
    /**
     Error: Device version is invalid.
    */
    GnaStatusDeviceVersionInvalid = -11,

    /**
     Error: Queue can not create or enqueue more requests.
    */
    GnaStatusDeviceQueueError = -12,

    /**
     Error: Failed to receive communication from the device driver.
    */
    GnaStatusDeviceIngoingCommunicationError = -13,

    /**
     Error: Failed to sent communication to the device driver.
    */
    GnaStatusDeviceOutgoingCommunicationError = -14,
    /**
     Error: Hardware device parameter out of range error occurred.
    */
    GnaStatusDeviceParameterOutOfRange = -15,

    /**
     Error: Hardware device virtual address out of range error occurred.
    */
    GnaStatusDeviceVaOutOfRange = -16,

    /**
     Error: Hardware device unexpected completion occurred during PCIe operation.
    */
    GnaStatusDeviceUnexpectedCompletion = -17,

    /**
     Error: Hardware device DMA error occurred during PCIe operation.
    */
    GnaStatusDeviceDmaRequestError = -18,

    /**
     Error: Hardware device MMU error occurred during PCIe operation.

    */
    GnaStatusDeviceMmuRequestError = -19,

    /**
     Error: Hardware device break-point hit.
    */
    GnaStatusDeviceBreakPointHit = -20,

    /**
     Error: Critical hardware device error occurred, device has been reset.
    */
    GnaStatusDeviceCriticalFailure = -21,

    /**
     Error: Memory buffer alignment is invalid.
    */
    GnaStatusMemoryAlignmentInvalid = -22,

    /**
     Error: Memory buffer size is invalid.
    */
    GnaStatusMemorySizeInvalid = -23,

    /**
     Error: Model total memory size exceeded.
    */
    GnaStatusMemoryTotalSizeExceeded = -24,

    /**
     Error: Memory buffer is invalid.
     E.g. outside of allocated memory or already released.
    */
    GnaStatusMemoryBufferInvalid = -25,

    /**
     Error: Waiting for a request failed.
    */
    GnaStatusRequestWaitError = -26,

     /**
     Error: Invalid number of active indices.
    */
    GnaStatusActiveListIndicesInvalid = -27,

    /**
     Error: Acceleration mode is not supported on this computer.
    */
    GnaStatusAccelerationModeNotSupported = -28,
};

 /**
 Gets message with detailed description of given status.

 @note
 TODO:3:API: provide maximum message size

 @param status The status code returned from API function.
 @param [out] messageBuffer User allocated buffer for the message.
 @param [in] messageBufferSize The size of the messageBuffer in bytes.
        The message is maximum X characters/bytes long.
        Message is truncated to messageBufferSize if it is longer than messageBufferSize characters.
 @return Status of fetching the message.
    @retval GnaStatusSuccess The status was fully serialized into the messageBuffer.
    @retval GnaStatusUnknownError The messageBuffer is too small. Message was truncated.
    @retval GnaNullargnotallowed The messageBuffer was NULL or messageBufferSize was 0.
 */
GNA_API enum GnaStatus GnaStatusGetMessage(enum GnaStatus status,
    char * messageBuffer, uint32_t messageBufferSize);

/**
 Rounds a number up, to the lowest multiple of significance.

 The function rounds the number up to the lowest possible value divisible
 by "significance".
 Used for calculating the memory sizes for GNA data buffers.

 @param number Memory size or a number to round up.
 @param significance The number that rounded value have to be divisible by.
 @return Rounded integer value.
 */

inline uint32_t GnaRoundUp(uint32_t number, uint32_t significance)
{
    return ((uint32_t)((number)+significance - 1) / significance) * significance;
};

/**
 Rounds a number up, to the lowest multiple of 64.
 @see GnaRoundUp().

 @param number Memory size or a number to round up.
 @return Rounded integer value.
 */
inline uint32_t GnaRoundUpTo64(uint32_t number)
{
    return GnaRoundUp(number, 64);
};

/**
 Definition of callback that is used to allocate "user owned" memory for model definition.

 Used for allocating "non-GNA" memory buffers used for model export or data-flow model
 structures (not model data).

 @warning
    User is responsible for releasing allocated memory buffers.

 @param size The size of the buffer to allocate, provided by GNA library.
 @return Allocated buffer.
 @retval NULL in case of allocation failure
 */
typedef void* (*GnaUserAllocator)(uint32_t size);

#endif //ifndef __GNA2_COMMON_API_H

/**
 @}
 @}
*/
