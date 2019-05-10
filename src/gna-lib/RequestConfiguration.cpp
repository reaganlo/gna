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

#include "RequestConfiguration.h"

#include <memory>

#include "Capabilities.h"
#include "CompiledModel.h"
#include "Expect.h"
#include "GnaException.h"
#include "GnaConfig.h"
#include "HardwareCapabilities.h"
#include "LayerConfiguration.h"


using namespace GNA;

RequestConfiguration::RequestConfiguration(CompiledModel& model, gna_request_cfg_id configId,
    DeviceVersion consistentDeviceIn) :
    Model{model},
    Id{configId},
    BufferElementCount{},
    consistentDevice{consistentDeviceIn}
{
    // TODO:3: optimize and store precalculated values if applicable for all layers
    HardwareCapabilities::GetHardwareConsistencySettings(BufferElementCount, consistentDevice);
}

void RequestConfiguration::AddBuffer(GnaComponentType type, uint32_t layerIndex, void *address)
{
    Expect::InRange(type, ComponentTypeCount,  Gna2StatusXnnErrorLyrCfg);
    Expect::NotNull(address);

    auto found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();

    auto emplaced = layerConfiguration->Buffers.emplace(type, address);
    Expect::True(emplaced.second, Gna2StatusIdentifierInvalid);

    auto layer = Model.GetLayer(layerIndex);
    auto bufferSize = layer->GetOperandSize(type);
    addMemoryObject(address, bufferSize);

    Model.InvalidateConfig(Id, layerConfiguration, layerIndex);
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, const ActiveList& activeList)
{
    const auto& layer = *Model.GetLayer(layerIndex);
    auto operation = layer.Operation;

    Expect::InSet(operation, { INTEL_AFFINE, INTEL_GMM }, Gna2StatusXnnErrorLyrOperation);

    auto found = LayerConfigurations.emplace(
        layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();

    Expect::Null(layerConfiguration->ActList.get());

    addMemoryObject((void *)activeList.Indices, activeList.IndicesCount * sizeof(uint32_t));

    auto activeListPtr = ActiveList::Create(activeList);
    layerConfiguration->ActList.swap(activeListPtr);
    ++ActiveListCount;

    Model.InvalidateConfig(Id, layerConfiguration, layerIndex);
}

void RequestConfiguration::SetHardwareConsistency(
    DeviceVersion consistentDeviceIn)
{
    if (Gna2DeviceVersionSoftwareEmulation != consistentDevice)
    {
        HardwareCapabilities::GetHardwareConsistencySettings(BufferElementCount, consistentDeviceIn);
        Acceleration.EnableHwConsistency();
        consistentDevice = consistentDeviceIn;
    }
    else
    {
        Acceleration.DisableHwConsistency();
    }
}

DeviceVersion RequestConfiguration::GetConsistentDevice() const
{
    return consistentDevice;
}

void RequestConfiguration::addMemoryObject(void *buffer, uint32_t bufferSize)
{
    auto memory = Model.FindBuffer(buffer, bufferSize);
    Expect::NotNull(memory, Gna2StatusIdentifierInvalid);

    if (!Model.IsPartOfModel(memory))
    {
        Model.ValidateBuffer(MemoryList, memory);
        MemoryList.push_back(memory);
    }
}
