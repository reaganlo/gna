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

#include "DeviceManager.h"

#if HW_VERBOSE == 1
#include "DeviceVerbose.h"
#endif

#include "Expect.h"
#include "Macros.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace GNA;

constexpr uint32_t DeviceManager::DefaultThreadCount;

DeviceManager::DeviceManager()
{
    // TODO:3: use DriverInterface method to determine number of devices
}

Device& DeviceManager::GetDevice(gna_device_id deviceId)
{
    try
    {
        if (deviceMap.empty())
        {
            // TODO:3: support multiple devices
            deviceMap.emplace_back(
#if HW_VERBOSE == 0
                std::make_unique<Device>(DeviceManager::DefaultThreadCount)
#else
                std::make_unique<DeviceVerbose>(DeviceManager::DefaultThreadCount)
#endif
            );
            deviceOpenedMap.emplace_back(false);
        }

        auto& device = *deviceMap.at(deviceId);
        return device;
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

uint32_t DeviceManager::GetDeviceCount()
{
    return 1;
}

DeviceVersion DeviceManager::GetDeviceVersion(gna_device_id deviceId)
{
    VerifyDeviceIndex(deviceId);

    const auto& device = GetDevice(deviceId);
    return device.GetVersion();
}

void DeviceManager::SetThreadCount(gna_device_id deviceId, uint32_t threadCount)
{
    VerifyDeviceIndex(deviceId);
    Expect::InRange(static_cast<uint32_t>(threadCount), 1U, 127U, Gna2StatusDeviceNumberOfThreadsInvalid);

    auto& device = GetDevice(deviceId);
    device.SetNumberOfThreads(threadCount);
}

uint32_t DeviceManager::GetThreadCount(gna_device_id deviceId)
{
    VerifyDeviceIndex(deviceId);

    const auto& device = GetDevice(deviceId);
    return device.GetNumberOfThreads();
}

void DeviceManager::VerifyDeviceIndex(gna_device_id deviceId)
{
    Expect::InRange(deviceId, ui32_0, GetDeviceCount() - 1, Gna2StatusIdentifierInvalid);
}

void DeviceManager::OpenDevice(gna_device_id deviceId)
{
    VerifyDeviceIndex(deviceId);

    GetDevice(deviceId);

    if (deviceOpenedMap.at(deviceId))
    {
        Log->Error("GNA Device already opened. Close Device first.\n");
        throw GnaException(Gna2StatusIdentifierInvalid);
    }

    deviceOpenedMap[deviceId] = true;
}

void DeviceManager::CloseDevice(gna_device_id deviceId)
{
    VerifyDeviceIndex(deviceId);

    if (!deviceOpenedMap.at(deviceId))
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }

    deviceOpenedMap[deviceId] = false;
}
