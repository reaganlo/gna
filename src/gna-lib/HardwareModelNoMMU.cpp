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

#define NOMINMAX 1

#include "HardwareModelNoMMU.h"

#include "CompiledModel.h"
#include "Memory.h"
#include <AffineLayers.h>

using namespace GNA;

class MemoryOfUndefinedSize : public Memory
{
public:
    MemoryOfUndefinedSize(void* address) :
        Memory(address, 1, 1)
    {}
};

inline std::map<uint32_t, uint32_t> const & HardwareModelNoMMU::GetBarMap(Gna2DeviceVersion target)
{
    static const auto barMap = std::map<Gna2DeviceVersion, std::map<uint32_t, uint32_t>>{
        {Gna2DeviceVersionEmbedded3_1, {
            { Gna2MemoryTagReadWrite, 1 },
            { Gna2MemoryTagInput, 2 },
            { Gna2MemoryTagOutput, 3 },
            { Gna2MemoryTagReadOnly, 1 },
            { Gna2MemoryTagScratch, 0 },
            { Gna2MemoryTagState, 3 },
        },},
        {Gna2DeviceVersionEmbedded3_5, {
            { Gna2MemoryTagReadWrite, 1 },
            { Gna2MemoryTagInput, 2 },
            { Gna2MemoryTagOutput, 3 },
            { Gna2MemoryTagReadOnly, 1 },
            { Gna2MemoryTagScratch, 0 },
            { Gna2MemoryTagState, 3 },
            { Gna2MemoryTagExternalBufferInput, std::numeric_limits<uint32_t>::max() },
            { Gna2MemoryTagExternalBufferOutput, std::numeric_limits<uint32_t>::max() },
        },},
    };

    try
    {
        return barMap.at(target);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }
}

inline uint32_t HardwareModelNoMMU::GetBarIndex(Gna2DeviceVersion target, uint32_t tag)
{
    try
    {
        return GetBarMap(target).at(tag);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusMemoryBufferInvalid);
    }
}

HardwareModelNoMMU::HardwareModelNoMMU(CompiledModel const & softwareModel, Gna2UserAllocator customAllocIn,
    Gna2DeviceVersion targetDevice) :
    HardwareModel(softwareModel, GetHwCaps(targetDevice)),
    customUserAlloc{ customAllocIn },
    exportAllocations{
    {Gna2MemoryTagReadWrite, {} },
    {Gna2MemoryTagInput, {} },
    {Gna2MemoryTagOutput, {} },
    {Gna2MemoryTagReadOnly, {} },
    {Gna2MemoryTagExternalBufferInput, {} },
    {Gna2MemoryTagExternalBufferOutput, {} },
    {Gna2MemoryTagScratch, {} },
    {Gna2MemoryTagState, {} },
}
{
    createScratchPadMemory(AffineBaseLayer::GetGlobal2MBScratchpad(), model.GetScratchpadSize());

    PrepareExportAllocations();

    // If there are regions tagged as External do not try guess
    if (exportAllocations.at(Gna2MemoryTagExternalBufferInput).empty() &&
        exportAllocations.at(Gna2MemoryTagExternalBufferOutput).empty() &&
        targetDevice == Gna2DeviceVersionEmbedded3_1)
    {
        GuessIOAllocations();
    }
}

void HardwareModelNoMMU::PrepareExportAllocations()
{
    auto const target = hwCapabilities.GetDeviceVersion();
    for (auto && buffer : model.GetAllocations())
    {
        auto tag = buffer->GetMemoryTag();
        auto const isTagDefined = GetBarMap(target).count(tag);
        auto const * memory = &(buffer.operator const Memory&());
        if (isTagDefined)
        {
            tag = (Gna2MemoryTagReadWrite == tag) ? Gna2MemoryTagReadOnly : tag;
            if (scratchPadMemory && buffer->GetBuffer() == scratchPadMemory->GetBuffer())
            {
                memory = scratchPadMemory.get();
            }
        }
        else // current implementation will treat all untagged buffers as RO
        {
            tag = Gna2MemoryTagReadOnly;
        }
        exportAllocations.at(tag).Emplace(*memory);
    }
}

void HardwareModelNoMMU::GuessIOAllocations()
{
    if (exportAllocations.at(Gna2MemoryTagInput).empty())
    {
        guessedInput = std::make_unique<MemoryOfUndefinedSize>(model.GetLayers().front()->Input.Buffer.Get());
        exportAllocations.at(Gna2MemoryTagInput).Emplace(*guessedInput);
    }
    if (exportAllocations.at(Gna2MemoryTagOutput).empty())
    {
        guessedOutput = std::make_unique<MemoryOfUndefinedSize>(model.GetLayers().back()->Output.Buffer.Get());
        exportAllocations.at(Gna2MemoryTagOutput).Emplace(*guessedOutput);
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
    auto const ldMemorySize = calculateDescriptorSize(false);

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
        {Gna2DeviceVersionEmbedded3_1, HardwareCapabilities{ Gna2DeviceVersionEmbedded3_1 }},
        {Gna2DeviceVersionEmbedded3_5, HardwareCapabilities{ Gna2DeviceVersionEmbedded3_5 }},
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
    if (!address)
        return 0;

    // TODO: 3: provide derived classes for 3.0 embedded and anna and using virtual methods to simplify the code,
    // TODO: 3: all first level IFs should be extracted IMO to dervied classes then
    auto const && found = std::find_if(exportAllocations.begin(), exportAllocations.end(),
        [&address](const auto & memories) { return memories.second.Contains(address); });
    if (exportAllocations.end() == found)
    {
        throw GnaException(Gna2StatusMemoryBufferInvalid);
    }

    auto const tag = found->first;
    auto const & memoryContainer = found->second;
    uint32_t offset;
    switch (tag)
    {
    case Gna2MemoryTagScratch:
        // Global scratchpad region starts after GnaDescriptor (32bytes) at BAR0
        offset = Gna2RoundUpTo64(GnaDescriptorSize) + memoryContainer.GetBufferOffset(address);
        break;
    case Gna2MemoryTagReadOnly:
        offset = ldMemory->GetSize() + memoryContainer.GetBufferOffset(address);
        break;
    case Gna2MemoryTagInput:
    case Gna2MemoryTagOutput:
    case Gna2MemoryTagState:
        offset = memoryContainer.GetBufferOffset(address);
        break;
    case Gna2MemoryTagExternalBufferInput:
    case Gna2MemoryTagExternalBufferOutput:
    {
        LdaOffset ldaOffset{ memoryContainer.GetBufferOffset(address) };
        ldaOffset.IsExternal = true;
        return ldaOffset;
    }
    default:
        throw GnaException(Gna2StatusMemoryBufferInvalid);
    }

    auto const target = this->hwCapabilities.GetDeviceVersion();
    return LdaOffset{ SetBarIndex(offset, GetBarIndex(target, tag)) };
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

    auto const & memoryContainer = GetComponent(component);
    const auto size = memoryContainer.GetMemorySize();
    exportDataSize = size;
    if (size == 0)
    {
        exportData = nullptr;
        return;
    }
    exportData = customAllocSafe(size);

    memoryContainer.CopyData(exportData, exportDataSize);
}

void HardwareModelNoMMU::ExportLd(void *& exportData, uint32_t & exportDataSize)
{
    Build({});

    exportData = ldMemory->GetBuffer();
    exportDataSize = ldMemory->GetSize();
}

MemoryContainer const& HardwareModelNoMMU::GetComponent(Gna2ModelExportComponent component) const
{
    auto const tag = GetComponentTag(hwCapabilities.GetDeviceVersion(), component);
    return exportAllocations.at(tag);
}

Gna2MemoryTag HardwareModelNoMMU::GetComponentTag(Gna2DeviceVersion target, Gna2ModelExportComponent component)
{
    static const auto mapping = std::map<Gna2DeviceVersion, std::map<Gna2ModelExportComponent, Gna2MemoryTag> >
    {
        { Gna2DeviceVersionEmbedded3_1,
            {
                { Gna2ModelExportComponentReadOnlyDump, Gna2MemoryTagReadOnly },
                { Gna2ModelExportComponentInputDump, Gna2MemoryTagInput },
                { Gna2ModelExportComponentOutputDump, Gna2MemoryTagOutput },
                { Gna2ModelExportComponentScratchDump, Gna2MemoryTagScratch },
            }
        },
        { Gna2DeviceVersionEmbedded3_5,
            {
                { Gna2ModelExportComponentReadOnlyDump, Gna2MemoryTagReadOnly },
                { Gna2ModelExportComponentInputDump, Gna2MemoryTagInput },
                { Gna2ModelExportComponentOutputDump, Gna2MemoryTagOutput },
                { Gna2ModelExportComponentScratchDump, Gna2MemoryTagScratch },
                { Gna2ModelExportComponentStateDump, Gna2MemoryTagState },
                { Gna2ModelExportComponentExternalBufferInputDump, Gna2MemoryTagExternalBufferInput },
                { Gna2ModelExportComponentExternalBufferOutputDump, Gna2MemoryTagExternalBufferOutput },
            }
        },
    };
    try
    {
        auto const & devMap = mapping.at(target);
        try
        {
            return devMap.at(component);
        }
        catch (std::out_of_range&)
        {
            throw GnaException(Gna2StatusMemoryBufferInvalid);
        }
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }
}
