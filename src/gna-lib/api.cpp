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

#include <memory>
#include <thread>

#include "gna-api-dumper.h"

#include "DeviceManager.h"
#include "Logger.h"
#include "Expect.h"

using namespace GNA;

static std::map<Gna2Status, gna_status_t> Gna2StatusToLegacy{
    {Gna2StatusWarningDeviceBusy, GNA_DEVICEBUSY},
    {Gna2StatusSuccess, GNA_SUCCESS},
    {Gna2StatusWarningArithmeticSaturation, GNA_SSATURATE},
    {Gna2StatusIdentifierInvalid, GNA_BADREQID} };

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
        auto& device = DeviceManager::Get().GetDevice(deviceId);
        device.LoadModel(modelId, model);
        return GNA_SUCCESS;
    }
    catch (const GnaModelException &e)
    {
        return e.getStatus();
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.ReleaseModel(modelId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.CreateConfiguration(modelId, configId);
        return GNA_SUCCESS;
    }
    catch (const GnaModelException &e)
    {
        return e.getStatus();
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.AttachBuffer(configId, type, layerIndex, address);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.AttachActiveList(configId, layerIndex, indicesCount, indices);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaRequestConfigEnableHardwareConsistency(
    gna_request_cfg_id configId,
    gna_device_version hardwareVersion)
{
    try
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.EnableHardwareConsistency(configId, (DeviceVersion)hardwareVersion);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);

    }
}

GNAAPI intel_gna_status_t GnaRequestConfigEnforceAcceleration(
    gna_request_cfg_id configId,
    gna_acceleration accel)
{
    try
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.EnforceAcceleration(configId, AccelerationMode(accel).GetMode());
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.ReleaseConfiguration(configId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.PropagateRequest(configId, requestId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        auto ret = device.WaitForRequest(requestId, milliseconds);
        return Gna2StatusToLegacy[ret];
    }
    catch (const GnaModelException &e)
    {
        return e.getStatus();
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        return CAST1_STATUS device.AllocateMemory(sizeRequested, sizeGranted, memoryAddress);
    }
    catch (const GnaException& e)
    {
        Log->Error(e.getStatus(), "Memory allocation failed.\n");
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        device.FreeMemory(memory);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        return e.getStatus();
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
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaDeviceSetThreadNumber(gna_device_id device, uint32_t threadNumber)
{
    try
    {
        DeviceManager::Get().SetThreadCount(device, threadNumber);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI intel_gna_status_t GnaDeviceOpen(gna_device_id device)
{
    try
    {
        DeviceManager::Get().OpenDevice(device);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}

GNAAPI gna_status_t GnaDeviceClose(
    gna_device_id deviceId)
{
    try
    {
        DeviceManager::Get().CloseDevice(deviceId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
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
        auto& device = DeviceManager::Get().GetDevice(0);
        return device.Dump(modelId, deviceGeneration, modelHeader, status, customAlloc);
    }
    catch (const GnaModelException &e)
    {
        *status = e.getStatus();
        return NULL;
    }
    catch (const GnaException &e)
    {
        *status = e.getStatus();
        return NULL;
    }
    catch (std::exception &e)
    {
        *status = HandleUnknownException(e);
        return NULL;
    }
}

gna_status_t GnaRequestConfigEnablePerf(gna_request_cfg_id configId,
        gna_hw_perf_encoding hwPerfEncoding, gna_perf_t* perfResults)
{
    try
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.EnableProfiling(configId, hwPerfEncoding, perfResults);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        return HandleUnknownException(e);
    }
}
