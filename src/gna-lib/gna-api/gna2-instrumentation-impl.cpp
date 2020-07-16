/*
 INTEL CONFIDENTIAL
 Copyright 2019-2020 Intel Corporation.

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

#include "gna2-common-impl.h"
#include "gna2-instrumentation-impl.h"

#include "Logger.h"
#include "Expect.h"
#include "DeviceManager.h"
#include "ApiWrapper.h"

using namespace GNA;

Gna2Status Gna2InstrumentationConfigSetMode(uint32_t configId,
    Gna2InstrumentationMode hwPerfEncoding)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.SetHardwareInstrumentation(configId, hwPerfEncoding);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigSetUnit(
    uint32_t configId,
    Gna2InstrumentationUnit instrumentationUnit)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.SetInstrumentationUnit(configId, instrumentationUnit);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t* results,
    uint32_t* configId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(configId);
        Expect::NotNull(selectedInstrumentationPoints);
        Expect::NotNull(results);
        Expect::GtZero(numberOfInstrumentationPoints, Gna2StatusIdentifierInvalid);
        auto& device = DeviceManager::Get().GetDevice(0);
        *configId = device.CreateProfilerConfiguration(
            { selectedInstrumentationPoints, selectedInstrumentationPoints + numberOfInstrumentationPoints },
            results);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigAssignToRequestConfig(
    uint32_t instrumentationConfigId,
    uint32_t requestConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.AssignProfilerConfigToRequestConfig(instrumentationConfigId, requestConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2Status Gna2InstrumentationConfigRelease(uint32_t instrumentationConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(0);
        device.ReleaseProfilerConfiguration(instrumentationConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}