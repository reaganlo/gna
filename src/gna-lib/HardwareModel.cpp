/*
 INTEL CONFIDENTIAL
 Copyright 2018-2020 Intel Corporation.

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

#include "HardwareModel.h"

#include "ActivationFunction.h"
#include "common.h"
#include "CompiledModel.h"
#include "Expect.h"
#include "GnaConfig.h"
#include "GnaException.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "Layer.h"
#include "Memory.h"
#include "SubModel.h"
#include "TransformMap.h"

#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <algorithm>

using namespace GNA;

uint32_t HardwareModel::calculateDescriptorSize(bool includeGmms) const
{
    auto const gmmDescriptorsSizeTmp = includeGmms ? gmmDescriptorsSize : 0;

    return xnnDescriptorsSize + gmmDescriptorsSizeTmp;
}

HardwareModel::HardwareModel(CompiledModel const & softwareModel, const HardwareCapabilities& hwCaps) :
    model{ softwareModel },
    hwCapabilities{ hwCaps },
    gmmDescriptorsSize{ getGmmDescriptorsSize(model.GmmCount) },
    xnnDescriptorsSize{ getLayerDescriptorsSize(model.LayerCount, hwCapabilities.GetDeviceVersion()) },
    HwModule{ HwModuleInterface::Create("gna_hw") }
{
}

//TODO:3: Remove and use HardwareModel per SubModel
bool HardwareModel::IsSoftwareLayer(const std::vector<std::unique_ptr<SubModel>>& submodels, uint32_t layerIndex)
{
    for (const auto& subModel : submodels)
    {
        if (layerIndex >= subModel->LayerIndex && layerIndex < subModel->LayerIndex + subModel->GetLayerCount() &&
            subModel->Type == SubmodelType::Software)
        {
            return true;
        }
    }
    return false;
}

void HardwareModel::Build(const std::vector<std::unique_ptr<SubModel>>& submodels)
{
    prepareAllocationsAndModel();

    auto gmmDescriptor = AddrGmmCfg();
    if (0 != gmmDescriptorsSize)
    {
        gmmDescriptor = AddrGmmCfg(ldMemory->GetBuffer<uint8_t>() +
                LayerDescriptor::GetSize(model.LayerCount, hwCapabilities.GetDeviceVersion()));
    }
    auto layerDescriptor = LayerDescriptor(*baseDescriptor, gmmDescriptor,
        getHwOffsetFunction);
    auto i = uint32_t { 0 };
    for (auto const & layerIter : model.GetLayers())
    {
        try
        {
            auto const & layer = *layerIter;
            const auto parameters = DescriptorParameters{layer, layerDescriptor, *HwModule };
            if (IsSoftwareLayer(submodels, i))
            {
                hardwareLayers.push_back(nullptr);
            }
            else
            {
                hardwareLayers.push_back(HardwareLayer::Create(parameters));
            }
            if (INTEL_GMM == layer.Operation)
            {
                gmmDescriptor++;
            }
            layerDescriptor.Forward(gmmDescriptor);
            i++;
        }
        catch (const GnaException& e)
        {
            throw GnaModelException(e, i);
        }
        catch (...)
        {
            throw GnaModelException(GnaException(Gna2StatusXnnErrorLyrCfg), i);
        }
    }
}

HardwareLayer const & HardwareModel::GetLayer(uint32_t layerIndex) const
{
    auto const layer = TryGetLayer(layerIndex);
    if (nullptr != layer)
    {
        return *layer;
    }
    throw GnaException(Gna2StatusXnnErrorLyrCfg);
}

HardwareLayer const * HardwareModel::TryGetLayer(uint32_t layerIndex) const
{
    try
    {
        return hardwareLayers.at(layerIndex).get();
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

// TODO:3: throw exception if not found, but NULL in nnet should be handled
uint32_t HardwareModel::GetBufferOffset(const BaseAddress& address) const
{
    return allocations.GetBufferOffset(address, PAGE_SIZE);
}

uint32_t HardwareModel::getLayerDescriptorsSize(
    const uint32_t layerCount, const DeviceVersion deviceVersion)
{
    auto layerDescriptorsSizeTmp = LayerDescriptor::GetSize(layerCount, deviceVersion);
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(const uint32_t gmmLayersCount)
{
    auto const gmmDescriptorsSizeTmp = size_t{gmmLayersCount * sizeof(GMM_CONFIG)};
    return static_cast<uint32_t>(gmmDescriptorsSizeTmp);
}

void HardwareModel::prepareAllocationsAndModel()
{
    Expect::InRange(model.LayerCount, ui32_1, HardwareCapabilities::GetMaximumLayerCount(DefaultDeviceVersion),
        Gna2StatusXnnErrorNetLyrNo);
    auto ldMemorySize = calculateDescriptorSize(true);
    auto ldSize = LayerDescriptor::GetSize(1, hwCapabilities.GetDeviceVersion());

    ldMemory = std::make_unique<Memory>(ldMemorySize, ldSize);
    if (!ldMemory)
    {
        throw GnaException {Gna2StatusResourceAllocationError};
    }

    prepareBaseDescriptor();

    allocations.Append(model.GetAllocations());

    // TODO:3:Validation is not correct for embedded platforms, fixme
    auto const modelSize = allocations.GetMemorySizeAlignedToPage();
    Expect::InRange(modelSize, hwCapabilities.MaximumModelSize,
        Gna2StatusMemoryTotalSizeExceeded);

    getHwOffsetFunction = [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); };
}

void HardwareModel::prepareBaseDescriptor()
{
    baseDescriptor = std::make_unique<LayerDescriptor>(
        *ldMemory, ldMemory->GetBuffer(), hwCapabilities);
    if (!baseDescriptor)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    // make ensure it's first on a list
    allocations.Emplace(*ldMemory);
}
