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

#include "gna2-inference-impl.h"

#include "ApiWrapper.h"
#include "Logger.h"
#include "DeviceManager.h"
#include "Macros.h"
#include "ModelWrapper.h"

#include "gna2-common-impl.h"

#include <stdint.h>
#include <vector>


using namespace GNA;

GNA2_API enum Gna2Status Gna2RequestConfigCreate(
    uint32_t modelId,
    uint32_t * requestConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.CreateConfiguration(modelId, requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigSetOperandBuffer(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t operandIndex,
    void * address)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.AttachBuffer(requestConfigId, operandIndex, operationIndex, address);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableActiveList(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t numberOfIndices,
    uint32_t const * indices)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.AttachActiveList(requestConfigId, operationIndex, numberOfIndices, indices);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableHardwareConsistency(
    uint32_t requestConfigId,
    enum Gna2DeviceVersion deviceVersion)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.EnableHardwareConsistency(requestConfigId, deviceVersion);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigSetAccelerationMode(
    uint32_t requestConfigId,
    enum Gna2AccelerationMode accelerationMode)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.EnforceAcceleration(requestConfigId, accelerationMode);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestConfigRelease(
    uint32_t requestConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.ReleaseConfiguration(requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestEnqueue(
    uint32_t requestConfigId,
    uint32_t * requestId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.PropagateRequest(requestConfigId, requestId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2RequestWait(
    uint32_t requestId,
    uint32_t timeoutMilliseconds)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        return device.WaitForRequest(requestId, timeoutMilliseconds);
    };
    return ApiWrapper::ExecuteSafely(command);
}

AccelerationMode::AccelerationMode(Gna2AccelerationMode basicMode, bool hardwareConsistencyEnabled)
    :mode{ basicMode },
    hardwareConsistency{ hardwareConsistencyEnabled }
{
    enforceValidity();
}

AccelerationMode::AccelerationMode(gna_acceleration legacyMode)
{
    switch (legacyMode)
    {
    case GNA_HARDWARE:
        mode = Gna2AccelerationModeHardware;
        break;
    case GNA_AUTO:
        mode = Gna2AccelerationModeAuto;
        break;
    case GNA_SOFTWARE:
        mode = Gna2AccelerationModeSoftware;
        break;
    case GNA_GENERIC:
        mode = Gna2AccelerationModeGeneric;
        break;
    case GNA_SSE4_2:
        mode = Gna2AccelerationModeSse4x2;
        break;
    case GNA_AVX1:
        mode = Gna2AccelerationModeAvx1;
        break;
    case GNA_AVX2:
        mode = Gna2AccelerationModeAvx2;
        break;
    default:
        mode = Gna2AccelerationModeAuto;
    }
    enforceValidity();
}

bool AccelerationMode::IsHardwareEnforced() const
{
    return mode == Gna2AccelerationModeHardware;
}

bool AccelerationMode::IsSoftwareEnforced() const
{
    return mode == Gna2AccelerationModeSoftware ||
        mode == Gna2AccelerationModeGeneric ||
        mode == Gna2AccelerationModeSse4x2 ||
        mode == Gna2AccelerationModeAvx1 ||
        mode == Gna2AccelerationModeAvx2;
}

AccelerationMode AccelerationMode::GetEffectiveSoftwareAccelerationMode(
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerations) const
{
    if (mode == Gna2AccelerationModeHardware)
    {
        throw GnaException(Gna2StatusAccelerationModeNotSupported);
    }
    if (mode == Gna2AccelerationModeSoftware ||
        mode == Gna2AccelerationModeAuto)
    {
        //last is fastest
        return AccelerationMode{ supportedCpuAccelerations.back(), hardwareConsistency };
    }
    for(const auto& supported: supportedCpuAccelerations)
    {
        if(mode == supported)
        {
            return AccelerationMode{ supported, hardwareConsistency };
        }
    }
    throw GnaException(Gna2StatusAccelerationModeNotSupported);
}

void AccelerationMode::SetMode(Gna2AccelerationMode newMode)
{
    mode = newMode;
    enforceValidity();
}

void AccelerationMode::EnableHwConsistency()
{
    hardwareConsistency = true;
    enforceValidity();
}

void AccelerationMode::DisableHwConsistency()
{
    hardwareConsistency = false;
    enforceValidity();
}

static std::map<AccelerationMode, const char*> AccelerationModeNames{
    {AccelerationMode{ Gna2AccelerationModeHardware },"GNA_HW"},
    {AccelerationMode{ Gna2AccelerationModeAuto,true },"GNA_AUTO_SAT"},
    {AccelerationMode{ Gna2AccelerationModeAuto,false },"GNA_AUTO_FAST"},
    {AccelerationMode{ Gna2AccelerationModeSoftware,true },"GNA_SW_SAT"},
    {AccelerationMode{ Gna2AccelerationModeSoftware,false },"GNA_SW_FAST"},
    {AccelerationMode{ Gna2AccelerationModeGeneric,true },"GNA_GEN_SAT"},
    {AccelerationMode{ Gna2AccelerationModeGeneric,false },"GNA_GEN_FAST"},
    {AccelerationMode{ Gna2AccelerationModeSse4x2,true },"GNA_SSE4_2_SAT"},
    {AccelerationMode{ Gna2AccelerationModeSse4x2,false },"GNA_SSE4_2_FAST"},
    {AccelerationMode{ Gna2AccelerationModeAvx1, true },"GNA_AVX1_SAT"},
    {AccelerationMode{ Gna2AccelerationModeAvx1, false },"GNA_AVX1_FAST"},
    {AccelerationMode{ Gna2AccelerationModeAvx2,true },"GNA_AVX2_SAT"},
    {AccelerationMode{ Gna2AccelerationModeAvx2,false },"GNA_AVX2_FAST"},
};

const char* AccelerationMode::UNKNOWN_ACCELERATION_MODE_NAME = "GNA_UNKNOWN_ACCELERATION_MODE";

const char* AccelerationMode::GetName() const
{
    auto item = AccelerationModeNames.find(*this);
    if (item != AccelerationModeNames.end())
    {
        return item->second;
    }
    return UNKNOWN_ACCELERATION_MODE_NAME;
}

bool AccelerationMode::GetHwConsistency() const
{
    return hardwareConsistency;
}

Gna2AccelerationMode GNA::AccelerationMode::GetMode() const
{
    return mode;
}

void AccelerationMode::enforceValidity()
{
    if (mode == Gna2AccelerationModeHardware)
    {
        hardwareConsistency = true;
    }
}

bool AccelerationMode::operator<(const AccelerationMode& right) const
{
    auto ret = (mode < right.mode) || ((mode == right.mode) && hardwareConsistency && (!right.hardwareConsistency));
    return ret;
}
