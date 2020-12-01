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

#include "HardwareModelNoMMU.h"

#include "CompiledModel.h"
#include "Memory.h"
#include <AffineLayers.h>

using namespace GNA;

class MemoryOfUndefinedSize : public Memory
{
public:
    MemoryOfUndefinedSize(void* address) :
    Memory(address ,1, 1)
    {
    }
};

HardwareModelNoMMU::HardwareModelNoMMU(CompiledModel const & softwareModel, Gna2UserAllocator customAllocIn,
    Gna2DeviceVersion targetDevice) :
    HardwareModel(softwareModel, GetHwCaps(targetDevice)),
    customUserAlloc{ customAllocIn },
    devComponentToMem
    {
        { Gna2DeviceVersionEmbedded3_1,
            {
                { Gna2ModelExportComponentReadOnlyDump, &FollowingLdaAllocations },
                { Gna2ModelExportComponentStateDump, &StateAllocations },
                { Gna2ModelExportComponentScratchDump, &ScratchAllocations },
                { Gna2ModelExportComponentExternalBufferInputDump, &ExternalBufferInputAllocations },
                { Gna2ModelExportComponentExternalBufferOutputDump, &ExternalBufferOutputAllocations },
            }
        },
        { Gna2DeviceVersionEmbedded3_0,
            {
                { Gna2ModelExportComponentReadOnlyDump, &FollowingLdaAllocations },
                { Gna2ModelExportComponentInputDump, &InputAllocations},
                { Gna2ModelExportComponentOutputDump, &OutputAllocations},
                { Gna2ModelExportComponentScratchDump, &ScratchAllocations },
            }
        }
    }
{
    for(const auto& memElement : softwareModel.GetAllocations())
    {
        const Memory& buffer = memElement;
        switch (buffer.GetTag())
        {
        case MemoryTagInput:
            InputAllocations.Emplace(buffer);
            break;
        case MemoryTagOutput:
            OutputAllocations.Emplace(buffer);
            break;
        case MemoryTagExternalBufferInput:
            ExternalBufferInputAllocations.Emplace(buffer);
            break;
        case MemoryTagExternalBufferOutput:
            ExternalBufferOutputAllocations.Emplace(buffer);
            break;
        case MemoryTagScratch:
            ScratchAllocations.Emplace(buffer);
            break;
        case MemoryTagState:
            StateAllocations.Emplace(buffer);
            break;
        case MemoryTagReadOnly:
        case MemoryTagReadWrite:
        default:
            if(memElement.GetBuffer() == AffineBaseLayer::GetGlobal2MBScratchpad())
            {
                ScratchAllocations.Emplace(buffer);
            }
            else
            {
                FollowingLdaAllocations.Emplace(buffer);
            }
        }
    }
    // If there are regions tagged as External do not try guess
    if(!ExternalBufferInputAllocations.empty() ||
       !ExternalBufferOutputAllocations.empty())
    {
        return;
    }
    if(InputAllocations.empty() && targetDevice == Gna2DeviceVersionEmbedded3_0)
    {
        guessedInput = std::make_unique<MemoryOfUndefinedSize>(softwareModel.GetLayers().front()->Input.Buffer.Get());
        InputAllocations.Emplace(*guessedInput);
    }
    if (OutputAllocations.empty() && targetDevice == Gna2DeviceVersionEmbedded3_0)
    {
        guessedOutput = std::make_unique<MemoryOfUndefinedSize>(softwareModel.GetLayers().back()->Output.Buffer.Get());
        OutputAllocations.Emplace(*guessedOutput);
    }
}

const LayerDescriptor& HardwareModelNoMMU::GetDescriptor(uint32_t layerIndex) const
{
    return GetLayer(layerIndex).XnnDescriptor;
}

uint32_t HardwareModelNoMMU::GetOutputOffset(uint32_t layerIndex) const
{
    auto layer = GetLayer(layerIndex);
    return layer.GetLdOutputOffset() - GetDescriptor(0).GetOffset();
}

uint32_t HardwareModelNoMMU::GetInputOffset(uint32_t layerIndex) const
{
    auto layer = GetLayer(layerIndex);
    return layer.GetLdInputOffset() - GetDescriptor(0).GetOffset();
}

void HardwareModelNoMMU::prepareAllocationsAndModel()
{
    Expect::InRange(model.LayerCount, 1u, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);
    auto const ldMemorySize = HardwareModel::calculateDescriptorSize(false);

    // TODO: don't use custom alloc here, as ptr alignment is required
    ldMemory = std::make_unique<Memory>(customAllocSafe(ldMemorySize), ldMemorySize);

    memset(ldMemory->GetBuffer(), 0, ldMemorySize);

    prepareBaseDescriptor();

    getHwOffsetFunction = [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); };
}

const HardwareCapabilities& HardwareModelNoMMU::GetHwCaps(Gna2DeviceVersion targetDevice)
{
    static const std::map<Gna2DeviceVersion, HardwareCapabilities> hwCaps =
    {
        {Gna2DeviceVersionEmbedded3_0, HardwareCapabilities{ Gna2DeviceVersionEmbedded3_0 }},
        {Gna2DeviceVersionEmbedded3_1, HardwareCapabilities{ Gna2DeviceVersionEmbedded3_1 }},
    };
    auto const found = hwCaps.find(targetDevice);
    if (hwCaps.cend() == found)
    {
        throw GnaException(Gna2StatusDeviceNotAvailable);
    }
    return found->second;
}

uint32_t HardwareModelNoMMU::SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex)
{
    return offsetFromBar | barIndex;
}

LdaOffset HardwareModelNoMMU::GetBufferOffset(const BaseAddress& address) const
{
    // TODO: 3: provide derived classes for 3.0 embedded and anna and using virtual methods to simplify the code,
    // TODO: 3: all first level IFs should be extracted IMO to dervied classes then
    if (InputAllocations.Contains(address))
    {
        Expect::True(this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_0, Gna2StatusMemoryBufferInvalid);
        return SetBarIndex(InputAllocations.GetBufferOffset(address), BarIndexInput);
    }
    if (OutputAllocations.Contains(address))
    {
        Expect::True(this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_0, Gna2StatusMemoryBufferInvalid);
        return SetBarIndex(OutputAllocations.GetBufferOffset(address), BarIndexOutput);
    }
    if (StateAllocations.Contains(address))
    {
        Expect::True(this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_1, Gna2StatusMemoryBufferInvalid);
        return SetBarIndex(StateAllocations.GetBufferOffset(address), StateBarIndex);
    }
    if (ScratchAllocations.Contains(address))
    {
        if (this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_0)
        {
            // Global scratchpad region starts after GnaDescriptor (32bytes) at BAR0
            return SetBarIndex(GnaDescritorSize + ScratchAllocations.GetBufferOffset(address), BarIndexGnaDescriptor);
        }
        Expect::True(this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_1, Gna2StatusMemoryBufferInvalid);
        return SetBarIndex(ScratchAllocations.GetBufferOffset(address), ScratchBarIndex);
    }
    if (FollowingLdaAllocations.Contains(address))
    {
        return SetBarIndex(ldMemory->GetSize() + FollowingLdaAllocations.GetBufferOffset(address), BarIndexLda);
    }
    if (ExternalBufferInputAllocations.Contains(address))
    {
        Expect::True(this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_1, Gna2StatusMemoryBufferInvalid);
        LdaOffset out{ ExternalBufferInputAllocations.GetBufferOffset(address) };
        out.SetExternalInput();
        return out;
    }
    if (ExternalBufferOutputAllocations.Contains(address))
    {
        Expect::True(this->hwCapabilities.GetDeviceVersion() == Gna2DeviceVersionEmbedded3_1, Gna2StatusMemoryBufferInvalid);
        LdaOffset out{ ExternalBufferOutputAllocations.GetBufferOffset(address) };
        out.SetExternalOutput();
        return out;
    }
    if (!address)
    {
        return 0;
    }
    throw GnaException(Gna2StatusMemoryBufferInvalid);
}

void HardwareModelNoMMU::ExportLd(void *& exportData, uint32_t & exportDataSize)
{
    Build({});

    exportData = ldMemory->GetBuffer();
    exportDataSize = ldMemory->GetSize();
}

void HardwareModelNoMMU::ExportComponent(void *& exportData, uint32_t & exportDataSize, Gna2ModelExportComponent component)
{
    // TODO: 3: extract into 2 methods (e.g., GetComponent + ExportData)
    if (component == Gna2ModelExportComponentLayerDescriptors)
    {
        ExportLd(exportData, exportDataSize);
        Expect::NotNull(exportData);
        return;
    }

    const auto devFound = devComponentToMem.find(hwCapabilities.GetDeviceVersion());
    Expect::True(devFound != devComponentToMem.end(), Gna2StatusNotImplemented);

    const auto found = devFound->second.find(component);

    Expect::True(found != devFound->second.end(), Gna2StatusMemoryBufferInvalid);

    auto& chosenComponent = *found->second;

    const auto size = chosenComponent.GetMemorySize();
    exportDataSize = size;
    if (size == 0)
    {
        exportData = nullptr;
        return;
    }
    exportData = customAllocSafe(size);

    chosenComponent.CopyData(exportData, exportDataSize);
}
