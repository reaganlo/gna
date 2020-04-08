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

#include "HardwareModelScorable.h"

#include "CompiledModel.h"
#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "Macros.h"
#include "Memory.h"
#include "MemoryContainer.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

#include "common.h"
#include "gna-api-status.h"
#include "profiler.h"

#include <utility>

using namespace GNA;

HardwareModelScorable::HardwareModelScorable(CompiledModel const & softwareModel,
    DriverInterface &ddi, const HardwareCapabilities& hwCapsIn) :
    HardwareModel(softwareModel, hwCapsIn),
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

    auto const modelSize = allocations.GetMemorySizeAlignedToPage();
    offset = requestConfiguration.GetAllocations().GetBufferOffset(address, PAGE_SIZE, modelSize);
    Expect::GtZero(offset, Gna2StatusMemoryBufferInvalid);
    return offset;
}

void HardwareModelScorable::InvalidateConfig(uint32_t configId)
{
    if (hardwareRequests.find(configId) != hardwareRequests.end())
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
        throw GnaException(Gna2StatusXnnErrorNetLyrNo);
    }
    for (auto i = layerIndex; i < layerIndex + layerCount; i++)
    {
        Expect::NotNull(TryGetLayer(i), Gna2StatusXnnErrorNetLyrNo);
    }

    auto operationMode = xNN;

    auto const & layer = model.GetLayer(layerIndex);
    if (layer.Operation == INTEL_GMM
        && !hwCapabilities.IsLayerSupported(layer.Operation)
        && hwCapabilities.HasFeature(LegacyGMM))
    {
        Expect::InRange(layerCount, ui32_1, Gna2StatusXnnErrorNetLyrNo);
        operationMode = GMM;
    }

    Expect::InRange(layerCount,
        ui32_1, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);

    SoftwareModel::LogAcceleration(AccelerationMode{ Gna2AccelerationModeHardware,true });
    SoftwareModel::LogOperationMode(operationMode);

    auto configId = requestConfiguration.Id;
    HardwareRequest *hwRequest = nullptr;

    {
        std::lock_guard<std::mutex> lockGuard(hardwareRequestsLock);
        if (hardwareRequests.find(configId) == hardwareRequests.end())
        {
            auto const inserted = hardwareRequests.emplace(
                configId,
                std::make_unique<HardwareRequest>(*this, requestConfiguration, allocations));
            hwRequest = inserted.first->second.get();
        }
        else
        {
            hwRequest = hardwareRequests.at(configId).get();
        }
    }
    hwRequest->Update(layerIndex, layerCount, operationMode);

    profiler->Measure(Gna2InstrumentationPointLibExecution);

    auto const result = driverInterface.Submit(*hwRequest, profiler);

    if (profiler != nullptr)
    {
        profiler->AddResults(Gna2InstrumentationPointDrvPreprocessing, result.driverPerf.Preprocessing);
        profiler->AddResults(Gna2InstrumentationPointDrvProcessing, result.driverPerf.Processing);
        profiler->AddResults(Gna2InstrumentationPointDrvDeviceRequestCompleted, result.driverPerf.DeviceRequestCompleted);
        profiler->AddResults(Gna2InstrumentationPointDrvCompletion, result.driverPerf.Completion);

        profiler->AddResults(Gna2InstrumentationPointHwTotalCycles, result.hardwarePerf.total);
        profiler->AddResults(Gna2InstrumentationPointHwStallCycles, result.hardwarePerf.stall);
    }

    if (result.status != Gna2StatusSuccess && result.status != Gna2StatusWarningArithmeticSaturation)
    {
        throw GnaException(result.status);
    }

    return (Gna2StatusWarningArithmeticSaturation == result.status) ? 1 : 0;
}

void HardwareModelScorable::ValidateConfigBuffer(MemoryContainer const & requestAllocations,
    Memory const & bufferMemory) const
{
    auto configModelSize = allocations.GetMemorySizeAlignedToPage();
    configModelSize += requestAllocations.GetMemorySizeAlignedToPage();
    configModelSize += RoundUp(bufferMemory.GetSize(), PAGE_SIZE);

    Expect::InRange(configModelSize, HardwareCapabilities::MaximumModelSize,
        Gna2StatusMemoryTotalSizeExceeded);
}

void HardwareModelScorable::allocateLayerDescriptors()
{
    HardwareModel::allocateLayerDescriptors();
    ldMemory->Map(driverInterface);
}
