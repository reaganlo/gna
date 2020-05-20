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

#include "HardwareModelSue1.h"

#include "AffineLayers.h"
#include "CompiledModel.h"
#include "GnaException.h"
#include "HardwareLayer.h"
#include "LayerDescriptor.h"
#include "Layer.h"
#include "Memory.h"

#include "gna2-model-suecreek-header.h"

using namespace GNA;


HardwareCapabilities HardwareModelSue1::sueCapabilities = HardwareCapabilities{ Gna2DeviceVersionEmbedded1_0 };

HardwareModelSue1::HardwareModelSue1(CompiledModel const & softwareModel, Gna2UserAllocator customAllocIn) :
    HardwareModel(softwareModel, sueCapabilities),
    customAlloc{ customAllocIn }
{
}

const LayerDescriptor& HardwareModelSue1::GetDescriptor(uint32_t layerIndex) const
{
    return GetLayer(layerIndex).XnnDescriptor;
}

uint32_t HardwareModelSue1::GetOutputOffset(uint32_t layerIndex) const
{
    auto layer = GetLayer(layerIndex);
    return layer.GetLdOutputOffset() - GetDescriptor(0).GetOffset();
}

uint32_t HardwareModelSue1::GetInputOffset(uint32_t layerIndex) const
{
    auto layer = GetLayer(layerIndex);
    return layer.GetLdInputOffset() - GetDescriptor(0).GetOffset();
}

void HardwareModelSue1::prepareAllocationsAndModel()
{
    Expect::InRange(model.LayerCount, ui32_1, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);

    auto const ldMemorySize = RoundUp(calculateDescriptorSize(false), PAGE_SIZE);

    uint32_t scratchPadSize = 0;
    for (auto const & layer : model.GetLayers())
    {
        auto const scratchPad = layer->TryGetOperand(ScratchpadOperandIndex);
        if (scratchPad)
        {
            scratchPadSize += scratchPad->Size;
        }
    }

    auto const rw = model.GetAllocations()[0];
    totalModelSize = ldMemorySize + rw.GetSize() + scratchPadSize;

    MemoryContainerElement const * ro = nullptr;
    if (model.GetAllocations().size() >= 3)
    {
        ro = &model.GetAllocations()[1];
        totalModelSize += ro->GetSize();
    }

    exportMemory = customAlloc(totalModelSize);
    if (!exportMemory)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    memset(exportMemory, 0, totalModelSize);

    ldMemory = std::make_unique<Memory>(exportMemory, ldMemorySize);
    if (!ldMemory)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
    prepareBaseDescriptor();

    allocations.Emplace(rw);

    if (scratchPadSize > 0)
    {
        scratchPadMemory = std::make_unique<Memory>(
            static_cast<uint8_t*>(exportMemory) + allocations.GetMemorySize(), scratchPadSize);
        if (!scratchPadMemory)
        {
            throw GnaException{ Gna2StatusResourceAllocationError };
        }
        allocations.Emplace(*scratchPadMemory);
    }

    if (nullptr != ro)
    {
        allocations.Emplace(*ro);
    }

    Expect::InRange(totalModelSize, static_cast<uint32_t>(2 * 1024 * 1024),
                    Gna2StatusMemoryTotalSizeExceeded);
    getHwOffsetFunction = [this](const BaseAddress& buffer) { return GetBufferOffset(buffer); };
}

LdaOffset HardwareModelSue1::GetBufferOffset(const BaseAddress& address) const
{
    if (address == AffineBaseLayer::GetGlobal2MBScratchpad())
    {
        return static_cast<uint32_t>(scratchPadMemory->GetBuffer<uint8_t>() - static_cast<uint8_t*>(exportMemory));
    }
    return allocations.GetBufferOffset(address);
}

void * HardwareModelSue1::Export()
{
    Build({});

    // copying data..
    auto const & rw = allocations[1];
    auto const & ro = allocations.back();
    void * destination = static_cast<uint8_t*>(exportMemory) + ldMemory->GetSize();
    memcpy_s(destination, totalModelSize, rw.GetBuffer(), rw.GetSize());
    if (allocations.size() >= 3)
    {
        auto const roSize = ro.GetSize();
        destination = static_cast<uint8_t*>(exportMemory) + totalModelSize - roSize;
        memcpy_s(destination, totalModelSize, ro.GetBuffer(), roSize);
    }

    return exportMemory;
}

void HardwareModelSue1::PopulateHeader(Gna2ModelSueCreekHeader & modelHeader) const
{
    // TODO:3: review
    auto const &input = model.GetLayer(0).Input;
    auto const &output = model.GetLayer(model.LayerCount - 1).Output;
    uint32_t outputsOffset = GetOutputOffset(model.LayerCount - 1);
    uint32_t inputsOffset = GetInputOffset(0);
    modelHeader =
    {
        0,
        static_cast<uint32_t>(totalModelSize),
        1,
        model.LayerCount,
        input.Mode.Size,
        output.Mode.Size,
        input.Count,
        output.Count,
        inputsOffset,
        outputsOffset,
        0,
        0,
        0,
        {}
    };
}
