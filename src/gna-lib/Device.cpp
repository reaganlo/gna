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

using std::ofstream;
using std::make_shared;
using std::make_unique;
using std::move;

using namespace GNA;

Device::Device(gna_device_id* deviceId, uint8_t threadCount) :
    requestHandler{threadCount}
{
    Expect::NotNull(deviceId);

    id = static_cast<gna_device_id>(std::hash<std::thread::id>()(std::this_thread::get_id()));

    *deviceId = id;
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
    auto &model = modelContainer.GetModel(modelId);
    requestBuilder.CreateConfiguration(model, configId);
}

void Device::EnableProfiling(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding, gna_perf_t * perfResults)
{
    auto& requestConfiguration = requestBuilder.GetConfiguration(configId);    
    requestConfiguration.HwPerfEncoding = hwPerfEncoding;
    requestConfiguration.PerfResults = perfResults;
}

void Device::AttachActiveList(gna_request_cfg_id configId, uint16_t layerIndex, uint32_t indicesCount, const uint32_t* const indices)
{
    auto activeList = ActiveList{indicesCount, indices};
    requestBuilder.AttachActiveList(configId, layerIndex, activeList);
}

void Device::ValidateSession(gna_device_id deviceId) const
{
    Expect::True(id == deviceId, GNA_INVALIDHANDLE);
}

void * Device::AllocateMemory(const uint32_t requestedSize, uint32_t * const sizeGranted)
{
    Expect::NotNull(sizeGranted);
    *sizeGranted = 0;

    totalMemory = make_unique<Memory>(requestedSize, XNN_LAYERS_MAX_COUNT, GMM_LAYERS_MAX_COUNT);
    *sizeGranted = static_cast<uint32_t>(totalMemory->ModelSize);
    Expect::True(*sizeGranted >= requestedSize, GNA_ERR_RESOURCES);

    return totalMemory->GetUserBuffer<void>();
}

void Device::FreeMemory()
{
    totalMemory.reset();
}

void Device::ReleaseModel(gna_model_id modelId)
{
    modelContainer.DeallocateModel(modelId);
    if (accelerationDetector.IsHardwarePresent())
    {
        totalMemory->Unmap();
    }
}

void Device::LoadModel(gna_model_id *modelId, const gna_model *raw_model)
{
    try
    {
        modelContainer.AllocateModel(modelId, raw_model, *totalMemory, accelerationDetector);
    }
    catch (...)
    {
        modelContainer.DeallocateModel(*modelId);
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
