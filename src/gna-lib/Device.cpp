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

#include <memory>

#include "Device.h"
#include "GnaException.h"

using std::make_shared;
using std::move;

using namespace GNA;

void Device::AttachBuffer(gna_request_cfg_id configId, gna_buffer_type type, uint16_t layerIndex, void *address)
{
    requestBuilder.AttachBuffer(configId, type, layerIndex, address);
}

void Device::CreateConfiguration(gna_model_id modelId, gna_request_cfg_id *configId)
{
    requestBuilder.CreateConfiguration(modelId, configId);
}

void Device::AttachActiveList(gna_request_cfg_id configId, uint16_t layerIndex, uint32_t indicesCount, uint32_t *indices)
{
    requestBuilder.AttachActiveList(configId, layerIndex, indicesCount, indices);
}

bool Device::ValidateSession(gna_device_id deviceId) const
{
    return id == deviceId && opened;
}

// TODO: implement as c-tor and propagate for members
status_t Device::Open(gna_device_id *deviceId, uint8_t threadCount)
{
    if(nHandles > GNA_DEVICE_LIMIT || opened)
    {
        ERR("GNA Device already opened. Close Device first.\n");
        throw GnaException(GNA_INVALIDHANDLE);
    }

    // detect available cpu accelerations
    accelerationDetector.DetectAccelerations();

    acceleration fastestMode = accelerationDetector.GetFastestAcceleration();
    acceleratorController.CreateAccelerators(accelerationDetector.IsHardwarePresent(), fastestMode);

    requestHandler.Init(threadCount);

    nHandles++;
    id = static_cast<gna_device_id>(
        std::hash<std::thread::id>()(std::this_thread::get_id()));

    opened = true;

    *deviceId = id;

    return GNA_SUCCESS;
}

// TODO: implement as d-tor and propagate for members
void Device::Close()
{
    if (!opened)
    {
        throw GnaException(GNA_INVALIDHANDLE);
    }

    acceleratorController.ClearAccelerators();
    requestHandler.ClearRequests();

    nHandles--;
    id = GNA_DEVICE_INVALID;
    opened = false;
}

size_t Device::AllocateMemory(size_t requestedSize, void **buffer)
{
    size_t modelSize = modelCompiler.CalculateModelSize(requestedSize, XNN_LAYERS_MAX_COUNT, GMM_LAYERS_MAX_COUNT);
    userMemory = _gna_malloc(modelSize);
    //TODO:KJ: verify if memory is zeroed before use.

    size_t internalSize = modelSize - requestedSize;
    *buffer = reinterpret_cast<uint8_t*>(userMemory) + internalSize;

    return requestedSize;
}

void Device::FreeMemory()
{
    _gna_free(userMemory);
    userMemory = nullptr;
}

void Device::ReleaseModel(gna_model_id modelId)
{
    modelContainer.DeallocateModel(modelId);
}

void Device::LoadModel(gna_model_id *modelId, const gna_model *raw_model)
{
    modelContainer.AllocateModel(modelId, raw_model);

    auto &model = modelContainer.GetModel(*modelId);
    modelCompiler.CascadeCompile(model, accelerationDetector);
}

void Device::PropagateRequest(gna_request_cfg_id configId, acceleration accel, gna_request_id *requestId)
{
    auto profiler = std::make_unique<RequestProfiler>();

    auto& configuration = requestBuilder.GetConfiguration(configId);
    auto& model = modelContainer.GetModel(configuration.ModelId);
    auto callback = [&, accel](KernelBuffers *buffers, RequestProfiler *profilerPtr){ return acceleratorController.ScoreModel(model, configuration, accel, profilerPtr, buffers); };

    auto request = std::make_unique<Request>(callback, move(profiler));
    requestHandler.Enqueue(requestId, std::move(request));
}

status_t Device::WaitForRequest(gna_request_id requestId, gna_timeout milliseconds)
{
    return requestHandler.WaitFor(requestId, milliseconds);
}
