/*
 INTEL CONFIDENTIAL
 Copyright 2017-2020 Intel Corporation.

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

#include "DeviceController.h"

#include <iostream>
#include <stdexcept>

DeviceController::DeviceController()
{
    auto deviceVersion = Gna2DeviceVersion{};
    auto deviceNumber = 0u;
    auto status = Gna2DeviceGetCount(&deviceNumber);
    ThrowOnStatusUnsuccessful(status, "GnaDeviceGetCount failed");

    gnaHandle = 0;
    // open the device
    status = Gna2DeviceOpen(gnaHandle);
    ThrowOnStatusUnsuccessful(status, "GnaDeviceOpen failed");

    status = Gna2DeviceGetVersion(gnaHandle, &deviceVersion);
    ThrowOnStatusUnsuccessful(status, "GnaDeviceGetVersion failed");

    std::cout << "Device version: " << std::hex << deviceVersion << std::endl;
}

DeviceController::~DeviceController()
{
    Gna2DeviceClose(gnaHandle);
}

void SetupTemporalInputOutput(const Gna2Model* model, void * memory)
{
    for (uint32_t i = 0; i < model->NumberOfOperations; i++)
    {
        for (uint32_t j = 0; j < 2; j++)
        {
            if (model->Operations[i].Operands[j]->Data == nullptr)
            {
                // TODO: 4: remove temporal assignment when Gna2ModelCreate accepts such tensors
                const_cast<void*&>(model->Operations[i].Operands[j]->Data) = memory;
            }
        }
    }
}

void DeviceController::ModelCreate(const Gna2Model* model, uint32_t* modelId) const
{
    SetupTemporalInputOutput(model, gnaMemory);
    auto const status = Gna2ModelCreate(gnaHandle, model, modelId);
    ThrowOnStatusUnsuccessful(status, "Model create2 failed");
}

void DeviceController::ModelRelease(uint32_t modelId) const
{
    auto const status = Gna2ModelRelease(modelId);
    ThrowOnStatusUnsuccessful(status, "Model create failed");
}

uint8_t * DeviceController::Alloc(uint32_t sizeRequested, uint32_t * sizeGranted)
{
    auto const status = Gna2MemoryAlloc(sizeRequested, sizeGranted, &gnaMemory);
    if (!Gna2StatusIsSuccessful(status) || nullptr == gnaMemory || sizeRequested > *sizeGranted)
    {
        throw std::runtime_error("Gna2MemoryAlloc failed");
    }

    return static_cast<uint8_t *>(gnaMemory);
}

void DeviceController::Free(void *memory)
{
    auto const status = Gna2MemoryFree(memory);
    ThrowOnStatusUnsuccessful(status, "Config add failed");
}

uint32_t DeviceController::ConfigAdd(uint32_t modelId)
{
    uint32_t configId = 0;
    auto const status = Gna2RequestConfigCreate(modelId, &configId);
    ThrowOnStatusUnsuccessful(status, "Config add failed");

    return configId;
}

void DeviceController::BufferAdd(uint32_t configId, uint32_t operationIndex,
    uint32_t operandIndex, void * address)
{
    auto const status = Gna2RequestConfigSetOperandBuffer(configId, operationIndex, operandIndex, address);
    ThrowOnStatusUnsuccessful(status, "Buffer add failed");
}

void DeviceController::RequestSetAcceleration(uint32_t configId, Gna2AccelerationMode accel)
{
    auto const status = Gna2RequestConfigSetAccelerationMode(configId, accel);
    ThrowOnStatusUnsuccessful(status, "RequestSetAcceleration add failed");
}

void DeviceController::RequestSetConsistency(uint32_t configId, Gna2DeviceVersion version)
{
    auto const status = Gna2RequestConfigEnableHardwareConsistency(configId, version);
    ThrowOnStatusUnsuccessful(status, "RequestSetConsistency add failed");
}

void DeviceController::BufferAddIO(uint32_t configId, uint32_t outputOperationIndex, void* input, void* output)
{
    BufferAdd(configId, 0, InputOperandIndex, input);
    BufferAdd(configId, outputOperationIndex, OutputOperandIndex, output);
}

void DeviceController::ActiveListAdd(uint32_t configId,
    uint32_t layerIndex, uint32_t indicesCount, uint32_t* indices)
{
    auto const status = Gna2RequestConfigEnableActiveList(
        configId, layerIndex, indicesCount, indices);
    ThrowOnStatusUnsuccessful(status, "ActiveList add failed");
}

void DeviceController::RequestEnqueue(uint32_t configId, uint32_t * requestId)
{
    auto const status = Gna2RequestEnqueue(configId, requestId);
    ThrowOnStatusUnsuccessful(status, "Request enqueue failed");
}

void DeviceController::RequestWait(uint32_t requestId)
{
    auto const status = Gna2RequestWait(requestId, 5 * 60 * 1000);
    ThrowOnStatusUnsuccessful(status, "Request wait failed");
}

void DeviceController::ThrowOnStatusUnsuccessful(Gna2Status const status, char const* message)
{
    if (!Gna2StatusIsSuccessful(status))
    {
        throw std::runtime_error(message);
    }
}
