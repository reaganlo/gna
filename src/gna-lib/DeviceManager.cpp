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
#include "GnaException.h"
#include "Logger.h"
#if defined(_WIN32)
#include "WindowsDriverInterface.h"
#else // linux
#include "LinuxDriverInterface.h"
#endif

#include "common.h"
#include "gna2-common-api.h"

#include <memory>

using namespace GNA;

constexpr uint32_t DeviceManager::DefaultThreadCount;

DeviceManager::DeviceManager()
{
    for (uint8_t i = 0; i < DriverInterface::MAX_GNA_DEVICES; i++)
    {
        std::unique_ptr<DriverInterface> driverInterface =
        {
    #if defined(_WIN32)
            std::make_unique<WindowsDriverInterface>()
    #else // GNU/Linux / Android / ChromeOS
            std::make_unique<LinuxDriverInterface>()
    #endif
        };
        const auto success = driverInterface->OpenDevice(i);
        if (success ||
            i == 0)
        {
            auto caps = HardwareCapabilities{};
            caps.DiscoverHardware(*driverInterface);
            if (caps.IsHardwareSupported() ||
                i == 0)
            {
                capabilities.emplace(i, caps);
            }
        }
    }
}

Device& DeviceManager::GetDevice(uint32_t deviceIndex)
{
    auto& device = *GetDeviceContext(deviceIndex).Handle;
    return device;
}

void DeviceManager::CreateDevice(uint32_t deviceIndex)
{
    if (!IsOpened(deviceIndex))
    {
        devices.emplace(deviceIndex, DeviceContext{
#if HW_VERBOSE == 0
           std::make_unique<Device>(deviceIndex, DeviceManager::DefaultThreadCount),
#else
            std::make_unique<DeviceVerbose>(deviceIndex, DeviceManager::DefaultThreadCount),
#endif
        0});
    }
}

bool DeviceManager::IsOpened(uint32_t deviceIndex)
{
    return devices.end() != devices.find(deviceIndex);
}

DeviceManager::DeviceContext& DeviceManager::GetDeviceContext(uint32_t deviceIndex)
{
    try
    {
        auto & deviceContext = devices.at(deviceIndex);
        return deviceContext;
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

DeviceManager::DeviceContext::DeviceContext(std::unique_ptr<Device> handle, uint32_t referenceCount) :
    Handle{std::move(handle)},
    ReferenceCount{referenceCount}
{
}

uint32_t DeviceManager::GetDeviceCount() const
{
    return static_cast<uint32_t>(capabilities.size());
}

DeviceVersion DeviceManager::GetDeviceVersion(uint32_t deviceIndex)
{
    if (IsOpened(deviceIndex)) // fetch opened device version
    {
        const auto& device = GetDevice(deviceIndex);
        return device.GetVersion();
    }
    else
    {
        try // fetch not yet opened device version
        {
            return capabilities.at(deviceIndex).GetHardwareDeviceVersion();
        }
        catch (std::out_of_range&)
        {
            throw GnaException(Gna2StatusIdentifierInvalid);
        }
    }
}

void DeviceManager::SetThreadCount(uint32_t deviceIndex, uint32_t threadCount)
{
    auto& device = GetDevice(deviceIndex);
    device.SetNumberOfThreads(threadCount);
}

uint32_t DeviceManager::GetThreadCount(uint32_t deviceIndex)
{
    const auto& device = GetDevice(deviceIndex);
    return device.GetNumberOfThreads();
}

void DeviceManager::OpenDevice(uint32_t deviceIndex)
{
    Expect::InRange(deviceIndex, GetDeviceCount() - 1, Gna2StatusIdentifierInvalid);
    
    CreateDevice(deviceIndex);

    auto & deviceRefCount = GetDeviceContext(deviceIndex).ReferenceCount;
    if (MaximumReferenceCount == deviceRefCount)
    {
        throw GnaException(Gna2StatusDeviceNotAvailable);
    }
    deviceRefCount++;
    Log->Message("Device %u opened, active handles: %u\n",
        deviceIndex, deviceRefCount);
}

void DeviceManager::CloseDevice(uint32_t deviceIndex)
{
    auto & deviceRefCount = GetDeviceContext(deviceIndex).ReferenceCount;
    if (deviceRefCount > 0)

    {
        --deviceRefCount;
        Log->Message("Device %u closed, active handles: %u\n",
            deviceIndex, deviceRefCount);

        if (deviceRefCount == 0)
        {
            devices.erase(deviceIndex);
        }
    }
    else
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

Device & DeviceManager::GetDeviceForModel(uint32_t modelId)
{
    const auto device = TryGetDeviceForModel(modelId);
    Expect::NotNull(device, Gna2StatusIdentifierInvalid);
    return *device;
}

Device* DeviceManager::TryGetDeviceForModel(uint32_t modelId)
{
    for(const auto& device : devices)
    {
       if(device.second.Handle && device.second.Handle->HasModel(modelId))
       {
           return device.second.Handle.get();
       }
    }
    return nullptr;
}

void DeviceManager::FreeMemory(void * memory)
{
    for (const auto& device : devices)
    {
        if (device.second.Handle && device.second.Handle->HasMemory(memory))
        {
            device.second.Handle->FreeMemory(memory);
            return;
        }
    }
}

Device & DeviceManager::GetDeviceForRequestConfigId(uint32_t requestConfigId)
{
    const auto device = TryGetDeviceForRequestConfigId(requestConfigId);
    Expect::NotNull(device, Gna2StatusIdentifierInvalid);
    return *device;
}

Device* DeviceManager::TryGetDeviceForRequestConfigId(uint32_t requestConfigId)
{
    for (const auto& device : devices)
    {
        if (device.second.Handle && device.second.Handle->HasRequestConfigId(requestConfigId))
        {
            return device.second.Handle.get();
        }
    }
    return nullptr;
}

Device & DeviceManager::GetDeviceForRequestId(uint32_t requestId)
{
    for (const auto& device : devices)
    {
        if (device.second.Handle && device.second.Handle->HasRequestId(requestId))
        {
            return *device.second.Handle;
        }
    }
    throw GnaException(Gna2StatusIdentifierInvalid);
}
