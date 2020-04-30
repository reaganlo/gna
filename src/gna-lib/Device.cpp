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

#include "Device.h"
#if HW_VERBOSE == 1
#include "DeviceVerbose.h"
#endif

#include "ActiveList.h"
#include "Expect.h"
#include "GnaException.h"
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"

#if defined(_WIN32)
#include "WindowsDriverInterface.h"
#else // linux
#include "LinuxDriverInterface.h"
#endif

#include <algorithm>
#include <cstdint>
#include <memory>

using namespace GNA;

uint32_t Device::modelIdSequence = 0;

Device::Device(uint32_t deviceIndex, uint32_t threadCount) :
    driverInterface
    {
#if defined(_WIN32)
        std::make_unique<WindowsDriverInterface>()
#else // GNU/Linux / Android / ChromeOS
        std::make_unique<LinuxDriverInterface>()
#endif
    },
    requestHandler{ threadCount }
{
    const auto success = driverInterface->OpenDevice(deviceIndex);
    if (success)
    {
        hardwareCapabilities.DiscoverHardware(driverInterface->GetCapabilities());
    }
    accelerationDetector.SetHardwareAcceleration(
        hardwareCapabilities.IsHardwareSupported());
    accelerationDetector.PrintAllAccelerationModes();
}

DeviceVersion Device::GetVersion() const
{
    return hardwareCapabilities.GetHardwareDeviceVersion();
}

uint32_t Device::GetNumberOfThreads() const
{
    return requestHandler.GetNumberOfThreads();
}

void Device::SetNumberOfThreads(uint32_t threadCount)
{
    requestHandler.ChangeNumberOfThreads(threadCount);
}

void Device::AttachBuffer(uint32_t configId,
    uint32_t operandIndex, uint32_t layerIndex, void *address)
{
    Expect::NotNull(address);

    requestBuilder.AttachBuffer(configId, operandIndex, layerIndex, address);
}

void Device::CreateConfiguration(uint32_t modelId, uint32_t *configId)
{
    auto &model = *models.at(modelId);
    requestBuilder.CreateConfiguration(model, configId,
                    hardwareCapabilities.GetDeviceVersion());
}

void Device::ReleaseConfiguration(uint32_t configId)
{
    requestBuilder.ReleaseConfiguration(configId);
}

void Device::EnableHardwareConsistency(
    uint32_t configId, DeviceVersion deviceVersion)
{
    if (Gna2DeviceVersionSoftwareEmulation == deviceVersion)
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }

    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.SetHardwareConsistency(deviceVersion);
}

void Device::EnforceAcceleration(uint32_t configId, Gna2AccelerationMode accelMode)
{
    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.EnforceAcceleration(accelMode);
}

void Device::AttachActiveList(uint32_t configId, uint32_t layerIndex,
        uint32_t indicesCount, const uint32_t* const indices)
{
    Expect::NotNull(indices);

    auto activeList = ActiveList{ indicesCount, indices };
    requestBuilder.AttachActiveList(configId, layerIndex, activeList);
}

bool Device::HasRequestConfigId(uint32_t requestConfigId) const
{
    return requestBuilder.HasConfiguration(requestConfigId);
}

bool Device::HasRequestId(uint32_t requestId) const
{
    return requestHandler.HasRequest(requestId);
}

void Device::MapMemory(Memory & memoryObject)
{
    if (hardwareCapabilities.IsHardwareSupported())
    {
        memoryObject.Map(*driverInterface);
    }
}

void Device::UnMapMemory(Memory & memoryObject)
{
    if (hardwareCapabilities.IsHardwareSupported())
    {
        memoryObject.Unmap(*driverInterface);
    }
}

void Device::ReleaseModel(uint32_t const modelId)
{
    models.erase(modelId);
}

void Device::PropagateRequest(uint32_t configId, uint32_t *requestId)
{
    Expect::NotNull(requestId);

    auto request = requestBuilder.CreateRequest(configId);
    requestHandler.Enqueue(requestId, std::move(request));
}

Gna2Status Device::WaitForRequest(uint32_t requestId, uint32_t milliseconds)
{
    return requestHandler.WaitFor(requestId, milliseconds);
}

void Device::Stop()
{
    requestHandler.StopRequests();
}

void Device::SetInstrumentationUnit(uint32_t configId, Gna2InstrumentationUnit instrumentationUnit)
{
    auto& requestConfiguration = requestBuilder.GetProfilerConfiguration(configId);
    requestConfiguration.SetUnit(instrumentationUnit);
}

void Device::SetHardwareInstrumentation(uint32_t configId, Gna2InstrumentationMode instrumentationMode)
{
    ProfilerConfiguration::ExpectValid(instrumentationMode);
    if (instrumentationMode > Gna2InstrumentationModeWaitForMmuTranslation
        && !hardwareCapabilities.HasFeature(NewPerformanceCounters))
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }

    auto& requestConfiguration = requestBuilder.GetProfilerConfiguration(configId);
    requestConfiguration.SetHwPerfEncoding(instrumentationMode);
}

bool Device::HasModel(uint32_t modelId) const
{
    return models.count(modelId) > 0;
}

uint32_t Device::CreateProfilerConfiguration(
    uint32_t numberOfInstrumentationPoints,
    Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t* results)
{
    Expect::NotNull(selectedInstrumentationPoints);
    Expect::NotNull(results);

    return requestBuilder.CreateProfilerConfiguration(numberOfInstrumentationPoints, selectedInstrumentationPoints, results);
}

void Device::AssignProfilerConfigToRequestConfig(uint32_t instrumentationConfigId,
    uint32_t requestConfigId)
{
    auto& requestConfiguration = requestBuilder.GetConfiguration(requestConfigId);
    auto& profilerConfiguration = requestBuilder.GetProfilerConfiguration(instrumentationConfigId);
    requestConfiguration.AssignProfilerConfig(&profilerConfiguration);
}

void Device::ReleaseProfilerConfiguration(uint32_t configId)
{
    requestBuilder.ReleaseProfilerConfiguration(configId);
}
