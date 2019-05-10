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

#include "HardwareModel.h"

#include <algorithm>

#include "AccelerationDetector.h"
#include "HardwareLayer.h"
#include "LayerConfiguration.h"
#include "Macros.h"
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

using namespace GNA;

uint32_t HardwareModel::CalculateDescriptorSize(
        const uint32_t layerCount, const uint32_t gmmLayersCount)
{
    Expect::InRange(layerCount, ui32_1,
        HardwareCapabilities::GetMaximumLayerCount(DefaultDeviceVersion),
        Gna2StatusXnnErrorNetLyrNo);

    auto layerDescriptorsSizeTmp = getLayerDescriptorsSize(layerCount);
    auto gmmDescriptorsSizeTmp = getGmmDescriptorsSize(gmmLayersCount);

    return layerDescriptorsSizeTmp + gmmDescriptorsSizeTmp;
}

HardwareModel::HardwareModel(
    const std::vector<std::unique_ptr<Layer>>& layers, uint32_t gmmCount,
    const HardwareCapabilities& hwCapsIn) :
    softwareLayers{ layers },
    hwCapabilities{ hwCapsIn },
    gmmDescriptorsSize{ getGmmDescriptorsSize(gmmCount) },
    xnnDescriptorsSize{ getLayerDescriptorsSize(static_cast<uint32_t>(layers.size()),
                                                        hwCapabilities.GetDeviceVersion()) }
{
}

uint64_t HardwareModel::GetMemoryId(const BaseAddress& address) const
{
    auto foundIt = std::find_if(modelMemoryObjects.cbegin(), modelMemoryObjects.cend(),
        [&address] (Memory *memory)
        {
            return address.InRange(memory->GetBuffer(),
                                    static_cast<uint32_t>(memory->GetSize()));
        });

    if (foundIt != modelMemoryObjects.cend())
        throw GnaException { Gna2StatusUnknownError };

    return (*foundIt)->GetId();
}

void HardwareModel::Build(const std::vector<Memory* >& modelMemoryObjectsIn)
{
    modelMemoryObjects = modelMemoryObjectsIn;

    allocateLayerDescriptors();

    modelSize = ALIGN(ldMemory->GetSize(), PAGE_SIZE);
    for (const auto memory : modelMemoryObjectsIn)
    {
        modelSize += ALIGN(memory->GetSize(), PAGE_SIZE);
    }
    Expect::InRange(modelSize, HardwareCapabilities::MaximumModelSize,
                    Gna2StatusMemoryTotalSizeExceeded);

    auto gmmDescriptor = AddrGmmCfg();
    if (0 != gmmDescriptorsSize)
    {
        gmmDescriptor = AddrGmmCfg(ldMemory->GetBuffer<uint8_t>() +
                LayerDescriptor::GetSize(static_cast<uint32_t>(softwareLayers.size()),
                                         hwCapabilities.GetDeviceVersion()));
    }
    auto layerDescriptor = LayerDescriptor(*baseDescriptor, gmmDescriptor,
        [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); });
    auto i = uint32_t { 0 };
    for (auto& layer : softwareLayers)
    {
        try
        {
            const auto parameters = DescriptorParameters{layer.get(), layerDescriptor };
            hardwareLayers.push_back(HardwareLayer::Create(parameters));
            if (INTEL_GMM == layer->Operation)
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

// TODO:3: throw exception if not found, but NULL in nnet should be handled
uint32_t HardwareModel::GetBufferOffset(const BaseAddress& address) const
{
    if (address.InRange(ldMemory->GetBuffer(),
                        static_cast<uint32_t>(ldMemory->GetSize())))
    {
        return address.GetOffset(BaseAddress{ ldMemory->GetBuffer() });
    }

    auto offset = ALIGN(ldMemory->GetSize(), PAGE_SIZE);
    for (auto memory : modelMemoryObjects)
    {
        if (address.InRange(memory->GetBuffer(),
                            static_cast<uint32_t>(memory->GetSize())))
        {
            return offset + address.GetOffset(BaseAddress{memory->GetBuffer()});
        }

        offset += ALIGN(memory->GetSize(), PAGE_SIZE);
    }

    return 0;
}

uint32_t HardwareModel::getLayerDescriptorsSize(
    const uint32_t layerCount, const DeviceVersion hwId)
{
    auto layerDescriptorsSizeTmp = LayerDescriptor::GetSize(layerCount, hwId);
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(const uint32_t gmmLayersCount)
{
    auto gmmDescriptorsSizeTmp = size_t{gmmLayersCount * sizeof(GMM_CONFIG)};
    return static_cast<uint32_t>(gmmDescriptorsSizeTmp);
}

void HardwareModel::allocateLayerDescriptors()
{
    auto ldMemorySize = xnnDescriptorsSize + gmmDescriptorsSize;
    auto ldSize = LayerDescriptor::GetSize(1, hwCapabilities.GetDeviceVersion());
    ldMemory = std::make_unique<Memory>(ldMemorySize, ldSize);

    if (!ldMemory)
    {
        throw GnaException {Gna2StatusResourceAllocationError};
    }

    baseDescriptor = std::make_unique<LayerDescriptor>(
            *ldMemory, ldMemory->GetBuffer(), hwCapabilities);
    if (!baseDescriptor)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
}
