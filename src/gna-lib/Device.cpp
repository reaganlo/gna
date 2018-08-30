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

#include <iostream>
#include <fstream>
#include <memory>

#include "ActiveList.h"
#include "FakeDetector.h"
#include "Memory.h"
#include "RequestConfiguration.h"

#include "Validator.h"

#if defined(_WIN32)
#include "WindowsIoctlSender.h"
#else // linux
#include "LinuxIoctlSender.h"
#endif

using std::ofstream;
using std::make_shared;
using std::make_unique;
using std::move;

using namespace GNA;

Device::Device(gna_device_id* deviceId, uint8_t threadCount) :
    ioctlSender{
#if defined(_WIN32)
    std::make_unique<WindowsIoctlSender>()
#else // linux
    std::make_unique<LinuxIoctlSender>()
#endif
    },
    accelerationDetector{*ioctlSender},
    memoryObjects{ },
    requestHandler{ threadCount }
{
    Expect::NotNull(deviceId);

    id = static_cast<gna_device_id>(std::hash<std::thread::id>()(std::this_thread::get_id()));

    *deviceId = id;

    accelerationDetector.UpdateKernelsMap();
}

void Device::AttachBuffer(gna_request_cfg_id configId, gna_buffer_type type, uint16_t layerIndex, void *address)
{
    Expect::NotNull(address);

    requestBuilder.AttachBuffer(configId, type, layerIndex, address);
}

void Device::CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId)
{
    Expect::NotNull(configId);

    auto memory = memoryObjects.front().get();
    auto &model = memory->GetModel(modelId);
    requestBuilder.CreateConfiguration(model, configId);
}

void Device::EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults)
{
    Expect::NotNull(perfResults);

    if (hwPerfEncoding >= DESCRIPTOR_FETCH_TIME
        && !accelerationDetector.HasFeature(NewPerformanceCounters))
    {
        throw GnaException(GNA_CPUTYPENOTSUPPORTED);
    }

    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);
    requestConfiguration.HwPerfEncoding = hwPerfEncoding;
    requestConfiguration.PerfResults = perfResults;
}

void Device::AttachActiveList(gna_request_cfg_id configId, uint16_t layerIndex, uint32_t indicesCount, const uint32_t* const indices)
{
    Expect::NotNull(indices);

    auto activeList = ActiveList{ indicesCount, indices };
    requestBuilder.AttachActiveList(configId, layerIndex, activeList);
}

void Device::ValidateSession(gna_device_id deviceId) const
{
    Expect::True(id == deviceId, GNA_INVALIDHANDLE);
}

void * Device::AllocateMemory(const uint32_t requestedSize, const uint16_t layerCount, uint16_t gmmCount, uint32_t * const sizeGranted)
{
    Expect::NotNull(sizeGranted);
    *sizeGranted = 0;

    auto memoryObject = createMemoryObject(requestedSize, layerCount, gmmCount);

    if (accelerationDetector.IsHardwarePresent())
    {
        memoryObject->Map();
    }

    *sizeGranted = memoryObject->ModelSize;
    auto userBuffer = memoryObject->GetUserBuffer();

    memoryObjects.emplace_back(std::move(memoryObject));

    return userBuffer;
}

void Device::FreeMemory()
{
    for (auto& memoryObject : memoryObjects)
    {
        if (memoryObject)
        {
            for (auto it = memoryObject->Models.begin(); it != memoryObject->Models.end(); ++it)
            {
                requestHandler.CancelRequests(it->first);
            }
        }
    }
    memoryObjects.clear();
}

void Device::LoadModel(gna_model_id *modelId, const gna_model *rawModel)
{
    Expect::NotNull(modelId);
    Expect::NotNull(rawModel);

    *modelId = modelIdSequence++;

    // default for 1st multi model phase
    auto& memory = *memoryObjects.front();
    try
    {
        memory.AllocateModel(*modelId, rawModel, accelerationDetector);
    }
    catch (...)
    {
        memory.DeallocateModel(*modelId);
        throw;
    }
}

void Device::PropagateRequest(gna_request_cfg_id configId, acceleration accel, gna_request_id *requestId)
{
    Expect::NotNull(requestId);

    auto request = requestBuilder.CreateRequest(configId, accel);
    requestHandler.Enqueue(requestId, std::move(request));
}

status_t Device::WaitForRequest(gna_request_id requestId, gna_timeout milliseconds)
{
    return requestHandler.WaitFor(requestId, milliseconds);
}

std::unique_ptr<Memory> Device::createMemoryObject(const uint32_t requestedSize,
    const uint16_t layerCount, const uint16_t gmmCount)
{
    return std::make_unique<Memory>(requestedSize, layerCount, gmmCount, *ioctlSender);
}
