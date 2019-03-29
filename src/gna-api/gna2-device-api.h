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
 @file gna2-device-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_DEVICE_API Device API

 API for accessing and managing GNA hardware and software devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_DEVICE_API_H
#define __GNA2_DEVICE_API_H

#include "gna2-common-api.h"

#include <stdint.h>

/**
 Gets number of available GNA devices on this computer.

 If no hardware device is available device number is set to 1,
 as software device still can be used.
 @see GnaDeviceGetVersion() to determine version of available device.

 @param [out] deviceCount Number of available devices.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaDeviceGetCount(
    uint32_t * deviceCount);

/**
 Retrieves hardware device version.

 Devices are zero-based indexed.
 Select desired device providing deviceIndex from 0 to GnaDeviceGetCount() - 1.
 @see GnaDeviceGetCount().

 @param deviceIndex Index of queried device.
 @param [out] deviceVersion GnaDeviceVersion identifier.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaDeviceGetVersion(
    uint32_t deviceIndex,
    enum GnaDeviceVersion * deviceVersion);

/**
 Sets number of software worker threads for given device.

 @note
    Must be called synchronously before GnaDeviceOpen().

 Device indexes are zero-based.
 Select desired device providing deviceIndex from 0 to GnaDeviceGetCount() - 1.

 @param deviceIndex Index of the affected device.
 @param threadNumber Number of software worker threads [1,127]. Default is 1.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaDeviceSetNumberOfThreads(
    uint32_t deviceIndex,
    uint32_t numberOfThreads);

/**
 Opens and initializes GNA device for processing.

 Device indexes are zero-based.
 Select desired device providing deviceIndex from 0 to GnaDeviceGetCount - 1.
 If no hardware devices are available, software device can be still opened
 by setting deviceIndex to 0.

 @note
 - The device has to be closed after usage to prevent resource leakage.
 - Only 1 device can stay opened at a time for current release.

 @param deviceIndex Index of the device to be opened.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaDeviceOpen(
    uint32_t deviceIndex);

/**
 Closes GNA device and releases the corresponding resources.

 @param deviceIndex The device to be closed.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaDeviceClose(
    uint32_t deviceIndex);

#endif // __GNA2_DEVICE_API_H

/**
 @}
 @}
 */
