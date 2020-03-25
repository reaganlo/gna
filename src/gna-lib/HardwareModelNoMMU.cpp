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

#include "HardwareModelNoMMU.h"

#include "CompiledModel.h"
#include "Memory.h"

using namespace GNA;

HardwareCapabilities HardwareModelNoMMU::noMMUCapabilities = HardwareCapabilities{ Gna2DeviceVersionEmbedded3_0 };

class MemoryOfUndefinedSize : public Memory
{
public:
    MemoryOfUndefinedSize(void* address) :
    Memory(address ,1, 1)
    {
    }
};

HardwareModelNoMMU::HardwareModelNoMMU(CompiledModel const & softwareModel, intel_gna_alloc_cb customAllocIn) :
    HardwareModel(softwareModel, noMMUCapabilities),
    customAlloc{ customAllocIn }
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
        case MemoryTagReadOnly:
        default:
            ROAllocations.Emplace(buffer);
        }
    }
    if(InputAllocations.empty())
    {
        guessedInput = std::make_unique<MemoryOfUndefinedSize>(softwareModel.GetLayers().front()->Input.Buffer.Get());
        InputAllocations.Emplace(*guessedInput);
    }
    if (OutputAllocations.empty())
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

void HardwareModelNoMMU::allocateLayerDescriptors()
{
    Expect::InRange(model.LayerCount, ui32_1, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);
    auto const ldMemorySize = HardwareModel::calculateDescriptorSize(false);

    ldMemory = std::make_unique<Memory>(customAlloc(ldMemorySize), ldMemorySize);

    if (ldMemory->GetBuffer() == nullptr)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    memset(ldMemory->GetBuffer(), 0, ldMemorySize);

    prepareBaseDescriptor();
}

uint32_t HardwareModelNoMMU::SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex)
{
    return offsetFromBar | barIndex;
}

void *getGlobal2MBScratchpad();

uint32_t HardwareModelNoMMU::GetBufferOffset(const BaseAddress& address) const
{
    if (address == getGlobal2MBScratchpad())
    {
        // Global scratchpad region starts after GnaDescriptor (32bytes) at BAR0
        return SetBarIndex(GnaDescritorSize, BarIndexGnaBar);
    }
    if (InputAllocations.Contains(address))
    {
        return SetBarIndex(InputAllocations.GetBufferOffset(address), BarIndexInput);
    }
    if (OutputAllocations.Contains(address))
    {
        return SetBarIndex(OutputAllocations.GetBufferOffset(address), BarIndexOutput);
    }
    if (ROAllocations.Contains(address))
    {
        return SetBarIndex(ldMemory->GetSize() + ROAllocations.GetBufferOffset(address), BarIndexRo);
    }
    throw GnaException(Gna2StatusMemoryBufferInvalid);
}

void HardwareModelNoMMU::ExportLd(void *& exportData, uint32_t & exportDataSize)
{
    Build({});

    exportData = ldMemory->GetBuffer();
    exportDataSize = ldMemory->GetSize();
}
