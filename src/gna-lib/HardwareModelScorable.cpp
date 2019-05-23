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

#include "HardwareModelScorable.h"

#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "Macros.h"
#include "Memory.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

#include "common.h"
#include "gna-api-instrumentation.h"
#include "gna-api-status.h"
#include "profiler.h"

#include <utility>

using namespace GNA;

HardwareModelScorable::HardwareModelScorable(
    const std::vector<std::unique_ptr<Layer>>& layers, uint32_t gmmCount,
    DriverInterface &ddi, const HardwareCapabilities& hwCapsIn) :
    HardwareModel(layers, gmmCount, hwCapsIn),
    driverInterface{ ddi }
{
}

uint32_t HardwareModelScorable::GetBufferOffsetForConfiguration(
    const BaseAddress& address,
    const RequestConfiguration& requestConfiguration) const
{
    auto offset = HardwareModel::GetBufferOffset(address);
    if (offset != 0)
    {
        return offset;
    }

    offset = modelSize;
    // TODO:3: collect offsets during buffer validation
    for (auto memory : requestConfiguration.MemoryList)
    {
        if (address.InRange(memory->GetBuffer(),
                            static_cast<uint32_t>(memory->GetSize())))
        {
            return offset + address.GetOffset(BaseAddress{memory->GetBuffer()});
        }

        offset += ALIGN(memory->GetSize(), PAGE_SIZE);
    }

    throw GnaException(Gna2StatusUnknownError);
}

void HardwareModelScorable::InvalidateConfig(gna_request_cfg_id configId)
{
    if(hardwareRequests.find(configId) != hardwareRequests.end())
    {
        hardwareRequests[configId]->Invalidate();
    }
}

uint32_t HardwareModelScorable::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    UNREFERENCED_PARAMETER(buffers);

    if (layerIndex + layerCount > hardwareLayers.size())
    {
        throw GnaException(Gna2StatusUnknownError);
    }

    Expect::InRange(layerCount,
        ui32_0, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);

    auto operationMode = xNN;

    const auto& layer = *softwareLayers.at(layerIndex);
    if (layer.Operation == INTEL_GMM && layerCount == 1
            && !hwCapabilities.IsLayerSupported(layer.Operation)
            && hwCapabilities.HasFeature(LegacyGMM))
    {
        operationMode = GMM;
    }

    SoftwareModel::LogAcceleration(AccelerationMode{Gna2AccelerationModeHardware,true});
    SoftwareModel::LogOperationMode(operationMode);

    auto configId = requestConfiguration.Id;
    HardwareRequest *hwRequest = nullptr;

    if (hardwareRequests.find(configId) == hardwareRequests.end())
    {
        auto inserted = hardwareRequests.emplace(
            configId,
            std::make_unique<HardwareRequest>(*this, requestConfiguration, ldMemory.get(), modelMemoryObjects));
        hwRequest = inserted.first->second.get();
    }
    else
    {
        hwRequest = hardwareRequests.at(configId).get();
    }

    hwRequest->Update(layerIndex, layerCount, operationMode);

    auto result = driverInterface.Submit(*hwRequest, profiler);

    auto perfResults = requestConfiguration.PerfResults;
    if (perfResults != nullptr)
    {
        perfResults->drv.startHW += result.driverPerf.startHW;
        perfResults->drv.scoreHW += result.driverPerf.scoreHW;
        perfResults->drv.intProc += result.driverPerf.intProc;

        perfResults->hw.stall += result.hardwarePerf.stall;
        perfResults->hw.total += result.hardwarePerf.total;
    }

    if (result.status != Gna2StatusSuccess && result.status != Gna2StatusWarningArithmeticSaturation)
    {
        throw GnaException(result.status);
    }

    return (Gna2StatusWarningArithmeticSaturation == result.status) ? 1 : 0;
}

void HardwareModelScorable::ValidateConfigBuffer(
    std::vector<Memory *> configMemoryList, Memory *bufferMemory) const
{
    auto configModelSize = modelSize;
    for (const auto memory : configMemoryList)
    {
        configModelSize += ALIGN(memory->GetSize(), PAGE_SIZE);
    }

    configModelSize += ALIGN(bufferMemory->GetSize(), PAGE_SIZE);

    Expect::InRange(configModelSize, HardwareCapabilities::MaximumModelSize,
                    Gna2StatusMemoryTotalSizeExceeded);
}

void HardwareModelScorable::allocateLayerDescriptors()
{
    HardwareModel::allocateLayerDescriptors();
    ldMemory->Map(driverInterface);
}


