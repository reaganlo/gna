/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "Device.h"
#if HW_VERBOSE == 1
#include "DeviceVerbose.h"
#endif
#include "Logger.h"
#include "Expect.h"

using std::thread;
using std::unique_ptr;

using namespace GNA;

std::unique_ptr<Device> GnaDevice;

intel_gna_status_t HandleUnknownException(const std::exception& e)
{
    Log->Error("Unknown exception: ", e.what());
    return GNA_UNKNOWN_ERROR;
}

/******************************************************************************
 *
 * API routines implementation
 *
 *****************************************************************************/

GNAAPI intel_gna_status_t GnaModelCreate(
    const gna_device_id deviceId,
    const gna_model* const model,
    gna_model_id* const modelId)
{
    try
    {
        GnaDevice->ValidateSession(deviceId);
        GnaDevice->LoadModel(modelId, model);
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

GNAAPI intel_gna_status_t GnaRequestConfigCreate(
    const gna_model_id modelId,
    gna_request_cfg_id* const configId)
{
    try
    {
        GnaDevice->CreateConfiguration(modelId, configId);
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

GNAAPI intel_gna_status_t GnaRequestConfigBufferAdd(
    const gna_request_cfg_id configId,
    const GnaComponentType type,
    const uint32_t layerIndex,
    void* const address)
{
    try
    {
        GnaDevice->AttachBuffer(configId, type, layerIndex, address);
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

GNAAPI intel_gna_status_t GnaRequestConfigActiveListAdd(
    const gna_request_cfg_id configId,
    const uint32_t layerIndex,
    const uint32_t indicesCount,
    const uint32_t* const indices)
{
    try
    {
        GnaDevice->AttachActiveList(configId, layerIndex, indicesCount, indices);
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
        GnaDevice->SetHardwareConsistency(configId, hardwareVersion);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        Log->Error("Unknown exception: ", e.what());
        return GNA_UNKNOWN_ERROR;
    }
}

GNAAPI intel_gna_status_t GnaRequestConfigEnforceAcceleration(
    gna_request_cfg_id configId,
    gna_acceleration accel)
{
    try
    {
        GnaDevice->EnforceAcceleration(configId, static_cast<AccelerationMode>(accel));
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        Log->Error("Unknown exception: ", e.what());
        return GNA_UNKNOWN_ERROR;
    }
}

GNAAPI intel_gna_status_t GnaRequestConfigRelease(gna_request_cfg_id configId)
{
    try
    {
        GnaDevice->ReleaseConfiguration(configId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (const std::exception& e)
    {
        Log->Error("Unknown exception: ", e.what());
        return GNA_UNKNOWN_ERROR;
    }
}

GNAAPI intel_gna_status_t GnaRequestEnqueue(
    const gna_request_cfg_id configId,
    gna_request_id* const requestId)
{
    try
    {
        GnaDevice->PropagateRequest(configId, requestId);
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

GNAAPI intel_gna_status_t GnaRequestWait(
    const gna_request_id requestId,
    const gna_timeout milliseconds)
{
    try
    {
        return GnaDevice->WaitForRequest(requestId, milliseconds);
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
    const intel_gna_status_t status)
{
    return Logger::StatusToString(status);
}

GNAAPI void* GnaAlloc(
    const gna_device_id deviceId,
    const uint32_t sizeRequested,
    const uint16_t layerCount,
    const uint16_t gmmCount,
    uint32_t* const sizeGranted)
{
    try
    {
        GnaDevice->ValidateSession(deviceId);
        return GnaDevice->AllocateMemory(sizeRequested, layerCount, gmmCount, sizeGranted);
    }
    catch (const GnaException& e)
    {
        Log->Error(e.getStatus(), "Memory allocation failed.\n");
        return nullptr;
    }
    catch (const std::exception& e)
    {
        return nullptr;
    }
}

GNAAPI intel_gna_status_t GnaFree(
    const gna_device_id deviceId)
{
    try
    {
        GnaDevice->ValidateSession(deviceId);
        GnaDevice->FreeMemory();
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
        *deviceCount = DeviceManager::GetDeviceCount();
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
        *deviceVersion = DeviceManager::GetDeviceVersion(deviceIndex);
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
        DeviceManager::SetThreadCount(device, threadNumber);
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
    if(GnaDevice)
    {
        Log->Error("GNA Device already opened. Close Device first.\n");
        return GNA_INVALIDHANDLE;
    }
    try
    {
        auto threadCount = DeviceManager::GetThreadCount(device);

#if HW_VERBOSE == 1
        GnaDevice = std::make_unique<DeviceVerbose>(device, threadCount);
#else
        GnaDevice = std::make_unique<Device>(device, threadCount);
#endif
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

GNAAPI intel_gna_status_t GnaDeviceClose(
    const gna_device_id deviceId)
{
    try
    {
        GnaDevice->ValidateSession(deviceId);
        GnaDevice->Stop();
        GnaDevice.reset();
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
    intel_gna_status_t* status,
    intel_gna_alloc_cb customAlloc)
{
    try
    {
        return GnaDevice->Dump(modelId, deviceGeneration, modelHeader, status, customAlloc);
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

intel_gna_status_t GnaRequestConfigEnablePerf(gna_request_cfg_id configId, gna_hw_perf_encoding hwPerfEncoding,
    gna_perf_t* perfResults)
{
    try
    {
        GnaDevice->EnableProfiling(configId, hwPerfEncoding, perfResults);
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
