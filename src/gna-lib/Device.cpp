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

#if WINDOWS == 1
#include "WindowsIoctlSender.h"
#else // LINUX

#endif

using std::ofstream;
using std::make_shared;
using std::make_unique;
using std::move;

using namespace GNA;

Device::Device(gna_device_id* deviceId, uint8_t threadCount) :
    requestHandler{ threadCount },
    memoryObjects{ APP_MEMORIES_LIMIT },
    ioctlSender{ 
#if WINDOWS == 1
    std::make_unique<WindowsIoctlSender>()
#else // LINUX
     
#endif 
    },
    accelerationDetector{*ioctlSender}
{
    Expect::NotNull(deviceId);

    id = static_cast<gna_device_id>(std::hash<std::thread::id>()(std::this_thread::get_id()));

    *deviceId = id;

    accelerationDetector.UpdateKernelsMap();
}

Device::~Device()
{
    FreeMemory();
}

void Device::AttachBuffer(gna_request_cfg_id configId, gna_buffer_type type, uint16_t layerIndex, void *address)
{
    requestBuilder.AttachBuffer(configId, type, layerIndex, address);
}

void Device::CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId)
{
    auto memoryId = 0;
    auto memory = memoryObjects.at(memoryId).get();
    auto &model = memory->GetModel(modelId);
    requestBuilder.CreateConfiguration(model, configId);
}

void Device::EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults)
{
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

    auto memoryId = 0ui64;
    for (; memoryId < memoryObjects.size(); ++memoryId)
    {
        if (!memoryObjects.at(memoryId))
            break;
    }

    if (APP_MEMORIES_LIMIT == memoryId)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }

    auto memoryObject = createMemoryObject(memoryId, requestedSize, layerCount, gmmCount);
    memoryObjects[memoryId] = std::move(memoryObject);

    auto memory = memoryObjects.at(memoryId).get();
    if (accelerationDetector.IsHardwarePresent())
    {
        memory->Map();
    }

    *sizeGranted = memory->ModelSize;
    return memory->GetUserBuffer();
}

void Device::FreeMemory()
{    
    if (accelerationDetector.IsHardwarePresent())
    {
        for (auto& memoryObject : memoryObjects)
        {
            if (memoryObject)
            {
                memoryObject->Unmap();
            }
        }
    }
    memoryObjects.clear();
}

void Device::LoadModel(gna_model_id *modelId, const gna_model *raw_model)
{
    *modelId = modelIdSequence++;
    auto memoryId = 0; // default for 1st multi model phase
    auto& memory = *memoryObjects.at(memoryId);
    try
    {
        memory.AllocateModel(*modelId, raw_model, accelerationDetector);
    }
    catch (...)
    {
        memory.DeallocateModel(*modelId);
        throw;
    }
}

void Device::PropagateRequest(gna_request_cfg_id configId, acceleration accel, gna_request_id *requestId)
{
    auto request = requestBuilder.CreateRequest(configId, accel);
    requestHandler.Enqueue(requestId, std::move(request));
}

status_t Device::WaitForRequest(gna_request_id requestId, gna_timeout milliseconds)
{
    return requestHandler.WaitFor(requestId, milliseconds);
}

std::unique_ptr<Memory> Device::createMemoryObject(const uint64_t memoryId, const uint32_t requestedSize,
    const uint16_t layerCount, const uint16_t gmmCount)
{
    return std::make_unique<Memory>(memoryId, requestedSize, layerCount, gmmCount, *ioctlSender);
}
