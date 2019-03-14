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

#include <stdexcept>
#include <iostream>

#include "DeviceController.h"

DeviceController::DeviceController()
{
    gna_device_version deviceVersion;
    gna_device_id deviceNumber = 0;
    auto status = GnaDeviceGetCount(&deviceNumber);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("GnaDeviceGetCount failed");
    }
    gnaHandle = 0;
    // open the device
    status = GnaDeviceOpen(gnaHandle);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("GnaDeviceOpen failed");
    }

    status = GnaDeviceGetVersion(gnaHandle, &deviceVersion);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("GnaDeviceGetVersion failed");
    }

    std::cout << "Device version: " << std::hex << deviceVersion << std::endl;
}

DeviceController::~DeviceController()
{
    intel_gna_status_t status = GnaDeviceClose(gnaHandle);
    if (GNA_SUCCESS != status)
    {
        // TODO log it, dtor should not throw
    }
}

void DeviceController::ModelCreate(const gna_model * model, gna_model_id * modelId)
{
    intel_gna_status_t status = GnaModelCreate(gnaHandle, model, modelId);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Model create failed");
    }
}

uint8_t * DeviceController::Alloc(uint32_t sizeRequested, uint32_t * sizeGranted)
{
    void *memory;
    GnaAlloc(sizeRequested, sizeGranted, &memory);
    if (nullptr == memory)
    {
        throw std::runtime_error("GnaAlloc failed");
    }

    return static_cast<uint8_t *>(memory);
}

void DeviceController::Free(void *memory)
{
    intel_gna_status_t status = GnaFree(memory);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Config add failed");
    }
}

gna_request_cfg_id DeviceController::ConfigAdd(gna_model_id modelId)
{
    gna_request_cfg_id configId;
    intel_gna_status_t status = GnaRequestConfigCreate(modelId, &configId);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Config add failed");
    }

    return configId;
}

void DeviceController::BufferAdd(gna_request_cfg_id configId, GnaComponentType type, uint32_t layerIndex, void * address)
{
    intel_gna_status_t status = GnaRequestConfigBufferAdd(configId, type, layerIndex, address);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Buffer add failed");
    }
}

void DeviceController::RequestSetAcceleration(gna_request_cfg_id configId, gna_acceleration accel)
{
    intel_gna_status_t status = GnaRequestConfigEnforceAcceleration(configId, accel);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("RequestSetAcceleration add failed");
    }
}

void DeviceController::RequestSetConsistency(gna_request_cfg_id configId, gna_device_version version)
{
    intel_gna_status_t status = GnaRequestConfigEnableHardwareConsistency(configId, version);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("RequestSetConsistency add failed");
    }
}

void DeviceController::ActiveListAdd(gna_request_cfg_id configId, uint32_t layerIndex, uint32_t indicesCount, uint32_t* indices)
{
    intel_gna_status_t status = GnaRequestConfigActiveListAdd(configId, layerIndex, indicesCount, indices);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("ActiveList add failed");
    }
}

void DeviceController::RequestEnqueue(gna_request_cfg_id configId, gna_request_id * requestId)
{
    intel_gna_status_t status = GnaRequestEnqueue(configId, requestId);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Request enqueue failed");
    }
}

void DeviceController::RequestWait(gna_request_id requestId)
{
    intel_gna_status_t status = GnaRequestWait(requestId, 5 * 60 * 1000);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Request wait failed");
    }
}


#if HW_VERBOSE == 1
void DeviceController::AfterscoreDebug(gna_model_id modelId, uint32_t nActions, dbg_action *actions)
{
    intel_gna_status_t status = GnaModelSetAfterscoreScenario(modelId, nActions, actions);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Setting after score scenario failed");
    }
}

void DeviceController::PrescoreDebug(gna_model_id modelId, uint32_t nActions, dbg_action *actions)
{
    intel_gna_status_t status = GnaModelSetPrescoreScenario(modelId, nActions, actions);
    if (GNA_SUCCESS != status)
    {
        throw std::runtime_error("Setting pre score scenario failed");
    }
}
#endif
