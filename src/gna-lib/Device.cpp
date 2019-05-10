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

#include <iostream>
#include <fstream>
#include <memory>

#include "ActiveList.h"
#include "Macros.h"
#include "Memory.h"
#include "RequestConfiguration.h"

#include "Expect.h"

#if defined(_WIN32)
#include "WindowsDriverInterface.h"
#else // linux
#include "LinuxDriverInterface.h"
#endif

using namespace GNA;

Device::Device(uint32_t threadCount) :
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
    try
    {
        driverInterface->OpenDevice();
        hardwareCapabilities.DiscoverHardware(*driverInterface);
        accelerationDetector.SetHardwareAcceleration(
            hardwareCapabilities.IsHardwareSupported());
    }
    catch (GnaException &e)
    {
        if (e.GetStatus() != Gna2StatusDeviceNotAvailable)
        {
            throw;
        }
    }
}

DeviceVersion Device::GetVersion() const
{
    return hardwareCapabilities.IsHardwareSupported()
        ? hardwareCapabilities.GetDeviceVersion()
        : Gna2DeviceVersionSoftwareEmulation;
}

uint32_t Device::GetNumberOfThreads() const
{
    return requestHandler.GetNumberOfThreads();
}

void Device::SetNumberOfThreads(uint32_t threadCount)
{
    requestHandler.ChangeNumberOfThreads(threadCount);
}

void Device::AttachBuffer(gna_request_cfg_id configId,
    GnaComponentType type, uint32_t layerIndex, void *address)
{
    Expect::NotNull(address);

    requestBuilder.AttachBuffer(configId, type, layerIndex, address);
}

void Device::CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId)
{
    auto &model = *models.at(modelId);
    requestBuilder.CreateConfiguration(model, configId,
                    hardwareCapabilities.GetDeviceVersion());
}

void Device::ReleaseConfiguration(gna_request_cfg_id configId)
{
    requestBuilder.ReleaseConfiguration(configId);
}

void Device::EnableHardwareConsistency(gna_request_cfg_id configId,
                                    DeviceVersion hardwareVersion)
{
    if (Gna2DeviceVersionSoftwareEmulation == hardwareVersion)
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }

    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.SetHardwareConsistency(hardwareVersion);
}

void Device::EnforceAcceleration(gna_request_cfg_id configId, Gna2AccelerationMode accelMode)
{
    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.EnforceAcceleration(accelMode);
}

void Device::EnableProfiling(gna_request_cfg_id configId,
    gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults)
{
    Expect::NotNull(perfResults);

    if (hwPerfEncoding >= DESCRIPTOR_FETCH_TIME
        && !hardwareCapabilities.HasFeature(NewPerformanceCounters))
    {
        throw GnaException(Gna2StatusAccelerationModeNotSupported);
    }

    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.HwPerfEncoding = hwPerfEncoding;
    requestConfiguration.PerfResults = perfResults;
}

void Device::AttachActiveList(gna_request_cfg_id configId, uint32_t layerIndex,
        uint32_t indicesCount, const uint32_t* const indices)
{
    Expect::NotNull(indices);

    auto activeList = ActiveList{ indicesCount, indices };
    requestBuilder.AttachActiveList(configId, layerIndex, activeList);
}

Gna2Status Device::AllocateMemory(uint32_t requestedSize,
        uint32_t *sizeGranted, void **memoryAddress)
{
    Expect::NotNull(sizeGranted);
    *sizeGranted = 0;

    auto memoryObject = createMemoryObject(requestedSize);

    if (hardwareCapabilities.IsHardwareSupported())
    {
        memoryObject->Map(*driverInterface);
    }

    *memoryAddress = memoryObject->GetBuffer();
    *sizeGranted = (uint32_t)memoryObject->GetSize();
    memoryObjects.emplace_back(std::move(memoryObject));
    return Gna2StatusSuccess;
}

void Device::FreeMemory(void *buffer)
{
    Expect::NotNull(buffer);

    auto memoryIterator = std::find_if(memoryObjects.begin(), memoryObjects.end(),
        [buffer] (std::unique_ptr<Memory>& memory)
        {
            if (memory->GetBuffer() == buffer)
            {
                return true;
            }
            return false;
        });

    if (memoryIterator == memoryObjects.end())
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }

    // TODO:3: mechanism to detect if memory is used in some model
    memoryObjects.erase(memoryIterator);
}

void Device::ReleaseModel(gna_model_id const modelId)
{
    models.erase(modelId);
}

void Device::LoadModel(gna_model_id *modelId, const gna_model *userModel)
{
    Expect::NotNull(modelId);
    Expect::NotNull(userModel);
    Expect::NotNull(userModel->pLayers);

    auto compiledModel = std::make_unique<CompiledModel>(
            userModel, accelerationDetector, hardwareCapabilities, memoryObjects);

    if (!compiledModel)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }

    *modelId = modelIdSequence++;

    compiledModel->BuildHardwareModel(*driverInterface);
    models.emplace(*modelId, std::move(compiledModel));
}

void Device::PropagateRequest(gna_request_cfg_id configId, uint32_t *requestId)
{
    Expect::NotNull(requestId);

    auto request = requestBuilder.CreateRequest(configId);
    requestHandler.Enqueue(requestId, std::move(request));
}

Gna2Status Device::WaitForRequest(gna_request_id requestId, gna_timeout milliseconds)
{
    return requestHandler.WaitFor(requestId, milliseconds);
}

void Device::Stop()
{
    requestHandler.StopRequests();
}

std::unique_ptr<Memory> Device::createMemoryObject(uint32_t requestedSize)
{
    return std::make_unique<Memory>(requestedSize);
}
