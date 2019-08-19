/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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
#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"

#include "gna2-common-impl.h"
#include "gna-api-dumper.h"
#include "ModelWrapper.h"

#include <cstddef>
#include <memory>

using namespace GNA;

static intel_gna_status_t HandleUnknownException(const std::exception& e)
{
    Log->Error("Unknown exception: %s.", e.what());
    return GNA_UNKNOWN_ERROR;
}

/******************************************************************************
 *
 * API routines implementation
 *
 *****************************************************************************/

GNAAPI gna_status_t GnaModelCreate(
    gna_device_id deviceId,
    gna_model const *model,
    gna_model_id *modelId)
{
    try
    {
        Expect::NotNull(modelId);
        Expect::NotNull(model);
        auto& device = DeviceManager::Get().GetDevice(deviceId);
        *modelId = device.LoadModel(*model);
        return GNA_SUCCESS;
    }
    catch (const GnaModelException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaModelRelease(
    gna_model_id modelId)
{
    try
    {
        auto device = DeviceManager::Get().TryGetDeviceForModel(modelId);
        if(device != nullptr)
        {
            device->ReleaseModel(modelId);
        }
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaRequestConfigCreate(
    const gna_model_id modelId,
    gna_request_cfg_id* const configId)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForModel(modelId);
        device.CreateConfiguration(modelId, configId);
        return GNA_SUCCESS;
    }
    catch (const GnaModelException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaRequestConfigBufferAdd(
    gna_request_cfg_id configId,
    GnaComponentType type,
    uint32_t layerIndex,
    void *address)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(configId);
        device.AttachBuffer(configId, ModelWrapper::GetOperandIndex(type), layerIndex, address);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaRequestConfigActiveListAdd(
    gna_request_cfg_id const configId,
    uint32_t const layerIndex,
    uint32_t const indicesCount,
    uint32_t const *const indices)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(configId);
        device.AttachActiveList(configId, layerIndex, indicesCount, indices);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaRequestConfigEnableHardwareConsistency(
    gna_request_cfg_id configId,
    gna_device_version legacyVersion)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(configId);
        const auto deviceVersion = DeviceVersionMapInverted.at(legacyVersion);
        device.EnableHardwareConsistency(configId, deviceVersion);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);

    }
}

GNAAPI intel_gna_status_t GnaRequestConfigEnforceAcceleration(
    gna_request_cfg_id configId,
    gna_acceleration accelerationMode)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(configId);
        device.EnforceAcceleration(configId, AccelerationMode(accelerationMode).GetMode());
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaRequestConfigRelease(gna_request_cfg_id configId)
{
    try
    {
        auto device = DeviceManager::Get().TryGetDeviceForRequestConfigId(configId);
        if (device != nullptr)
        {
            device->ReleaseConfiguration(configId);
        }
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaRequestEnqueue(
    const gna_request_cfg_id configId,
    gna_request_id* const requestId)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestConfigId(configId);
        device.PropagateRequest(configId, requestId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaRequestWait(
    gna_request_id requestId,
    gna_timeout milliseconds)
{
    try
    {
        auto& device = DeviceManager::Get().GetDeviceForRequestId(requestId);
        auto status = device.WaitForRequest(requestId, milliseconds);
        return StatusMap.at(status);
    }
    catch (const GnaModelException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI char const * GnaStatusToString(
    const gna_status_t status)
{
    return Logger::StatusToString(status);
}

// TODO: support memory allocation for multiple devices
GNAAPI gna_status_t GnaAlloc(
    uint32_t const sizeRequested,
    uint32_t *const sizeGranted,
    void **memoryAddress)
{
    try
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        auto status = device.AllocateMemory(sizeRequested, sizeGranted, memoryAddress);
        return StatusMap.at(status);
    }
    catch (const GnaException& e)
    {
        Log->Error(e.GetLegacyStatus(), "Memory allocation failed.\n");
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaFree(
    void *memory)
{
    try
    {
        DeviceManager::Get().FreeMemory(memory);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaDeviceGetCount(uint32_t * deviceCount)
{
    try
    {
        Expect::NotNull(deviceCount);
        *deviceCount = DeviceManager::Get().GetDeviceCount();
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaDeviceGetVersion(uint32_t deviceIndex,
    gna_device_version * deviceVersion)
{
    try
    {
        Expect::NotNull(deviceVersion);
        *deviceVersion = (gna_device_version)DeviceManager::Get().GetDeviceVersion(deviceIndex);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaDeviceSetThreadNumber(gna_device_id deviceIndex, uint32_t threadNumber)
{
    try
    {
        DeviceManager::Get().SetThreadCount(deviceIndex, threadNumber);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaDeviceOpen(gna_device_id deviceIndex)
{
    try
    {
        DeviceManager::Get().OpenDevice(deviceIndex);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaDeviceClose(
    gna_device_id deviceIndex)
{
    try
    {
        DeviceManager::Get().CloseDevice(deviceIndex);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.GetLegacyStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

void* GnaModelDump(
    gna_model_id modelId,
    gna_device_generation deviceGeneration,
    intel_gna_model_header* modelHeader,
    gna_status_t* status,
    intel_gna_alloc_cb customAlloc)
{
    try
    {
        Gna2Status newStatus;
        auto& device = DeviceManager::Get().GetDevice(0);
        Expect::Equal(GNA_1_0_EMBEDDED, deviceGeneration, Gna2StatusAccelerationModeNotSupported); // Temporary limitation
        auto dump = device.Dump(modelId, modelHeader, &newStatus, customAlloc);
        *status = StatusMap.at(newStatus);
        return dump;
    }
    catch (const GnaModelException &e)
    {
        *status = e.GetLegacyStatus();
        return NULL;
    }
    catch (const GnaException &e)
    {
        *status = e.GetLegacyStatus();
        return NULL;
    }
    catch (std::exception &e)
    {
        *status = HandleUnknownException(e);
        return NULL;
    }
}

