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

#include "RequestConfiguration.h"

#include "ActiveList.h"
#include "Address.h"
#include "CompiledModel.h"
#include "Expect.h"
#include "HardwareCapabilities.h"
#include "KernelArguments.h"
#include "Layer.h"
#include "LayerConfiguration.h"

#include "gna-api-status.h"

#include <memory>
#include <utility>

using namespace GNA;

RequestConfiguration::RequestConfiguration(CompiledModel& model, gna_request_cfg_id configId,
    DeviceVersion consistentDeviceIn) :
    Model{ model },
    Id{ configId },
    BufferElementCount{ HardwareCapabilities::GetHardwareConsistencySettings(consistentDeviceIn) },
    consistentDevice{ consistentDeviceIn }
{
}

void RequestConfiguration::AddBuffer(uint32_t operandIndex, uint32_t layerIndex, void *address)
{
    auto context = AddBufferContext(Model, operandIndex, layerIndex, address);
    storeAllocationIfNew(context.Address, context.Size);

    if (ScratchpadOperandIndex == layerIndex)
    {
        addBufferForMultipleLayers(context);
    }
    else
    {
        addBufferForSingleLayer(context);

    }
    Model.InvalidateHardwareRequestConfig(Id);
}

RequestConfiguration::AddBufferContext::AddBufferContext(CompiledModel & model,
    uint32_t operandIndexIn, uint32_t layerIndexIn, void * addressIn) :
    SoftwareLayer{ nullptr },
    Operand{ nullptr },
    OperandIndex{ operandIndexIn },
    LayerIndex{ layerIndexIn },
    Address{ addressIn }
{
    Expect::NotNull(Address);

    if (ScratchpadOperandIndex == LayerIndex)
    {
        Size = model.GetMaximumOperandSize(OperandIndex);
    }
    else
    {
        SoftwareLayer = &model.GetLayer(LayerIndex);
        Operand = &SoftwareLayer->GetOperand(OperandIndex);
        Size = Operand->Size;
    }
}

void RequestConfiguration::addBufferForMultipleLayers(AddBufferContext & context)
{
    context.LayerIndex = 0;
    for (auto const & layerIter : Model.GetLayers())
    {
        context.SoftwareLayer = layerIter.get(); // not null assured
        context.Operand = context.SoftwareLayer->TryGetOperand(context.OperandIndex);

        applyBufferForSingleLayer(context);
        context.LayerIndex++;
    }
}

void RequestConfiguration::addBufferForSingleLayer(AddBufferContext & context)
{
    context.SoftwareLayer = &Model.GetLayer(context.LayerIndex);
    Expect::NotNull(context.Operand, Gna2StatusXnnErrorLyrCfg);
    applyBufferForSingleLayer(context);
}


void RequestConfiguration::applyBufferForSingleLayer(AddBufferContext & context)
{
    if (nullptr != context.Operand)
    {
        auto & layerConfiguration = getLayerConfiguration(context.LayerIndex);
        layerConfiguration.EmplaceBuffer(context.OperandIndex, context.Address);

        // if invalidate fails, we don't know if it's already been used thus no recovery from this
        context.SoftwareLayer->UpdateKernelConfigs(layerConfiguration);
    }
}

LayerConfiguration & RequestConfiguration::getLayerConfiguration(uint32_t layerIndex)
{
    auto const found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto & layerConfiguration = *found.first->second;
    return layerConfiguration;
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, const ActiveList& activeList)
{
    const auto& layer = Model.GetLayer(layerIndex);

    Expect::InSet(layer.Operation, { INTEL_AFFINE, INTEL_GMM }, Gna2StatusXnnErrorLyrOperation);

    auto & layerConfiguration = getLayerConfiguration(layerIndex);

    Expect::Null(layerConfiguration.ActList.get());

    storeAllocationIfNew(activeList.Indices,
        activeList.IndicesCount * static_cast<uint32_t>(sizeof(uint32_t)));

    auto activeListPtr = ActiveList::Create(activeList);
    layerConfiguration.ActList.swap(activeListPtr);
    ++ActiveListCount;

    layer.UpdateKernelConfigs(layerConfiguration);
    Model.InvalidateHardwareRequestConfig(Id);
}

void RequestConfiguration::SetHardwareConsistency(
    DeviceVersion consistentDeviceIn)
{
    if (Gna2DeviceVersionSoftwareEmulation != consistentDeviceIn)
    {
        Expect::True(Model.IsFullyHardwareCompatible(HardwareCapabilities{ consistentDeviceIn }), Gna2StatusAccelerationModeNotSupported);
        BufferElementCount = HardwareCapabilities::GetHardwareConsistencySettings(consistentDeviceIn);
        BufferElementCountForAdl = HardwareCapabilities::GetHardwareConsistencySettingsForAdl(consistentDeviceIn);
    }
    Acceleration.SetHwConsistency(Gna2DeviceVersionSoftwareEmulation != consistentDeviceIn);
    consistentDevice = consistentDeviceIn;
}

void RequestConfiguration::EnforceAcceleration(Gna2AccelerationMode accelMode)
{
    if (accelMode == Gna2AccelerationModeHardware)
    {
        Expect::True(Model.IsHardwareEnforcedModeValid(), Gna2StatusAccelerationModeNotSupported);
    }
    Acceleration.SetMode(accelMode);
}

DeviceVersion RequestConfiguration::GetConsistentDevice() const
{
    return consistentDevice;
}

uint8_t RequestConfiguration::GetHwInstrumentationMode() const
{
    if (profilerConfiguration != nullptr)
    {
        auto const encoding = static_cast<int>(
            profilerConfiguration->HwPerfEncoding) + 1;
        return static_cast<uint8_t>(encoding & 0xFF);
    }
    return 0;
}

void RequestConfiguration::storeAllocationIfNew(void const *buffer, uint32_t bufferSize)
{
    // add buffer memory if is not already included in model memory
    auto const memory = Model.GetMemoryIfNotPartOfModel(buffer, bufferSize);
    if (nullptr != memory)
    {
        Model.ValidateBuffer(allocations, *memory);
        allocations.Emplace(*memory);
    }
    // else buffer already in model memory
}
