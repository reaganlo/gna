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

#include "HardwareModelSue1.h"

#include "CompiledModel.h"
#include "GnaException.h"
#include "HardwareLayer.h"
#include "LayerDescriptor.h"
#include "Layer.h"
#include "Memory.h"

#include "gna-api-status.h"

#include <algorithm>

using namespace GNA;


HardwareCapabilities HardwareModelSue1::sueCapabilities = HardwareCapabilities{ Gna2DeviceVersionEmbedded1_0 };

HardwareModelSue1::HardwareModelSue1(CompiledModel const & softwareModel, intel_gna_alloc_cb customAllocIn) :
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

void HardwareModelSue1::allocateLayerDescriptors()
{
    Expect::InRange(model.LayerCount, ui32_1, hwCapabilities.GetMaximumLayerCount(),
        Gna2StatusXnnErrorNetLyrNo);
    auto const ldMemorySize = RoundUp(HardwareModel::calculateDescriptorSize(false), PAGE_SIZE);
    auto const modelSize = model.GetSize();
    totalModelSize = ldMemorySize + modelSize;

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
}

uint32_t HardwareModelSue1::GetBufferOffset(const BaseAddress& address) const
{
    return allocations.GetBufferOffset(address);
}

void * HardwareModelSue1::Export()
{
    Build({});

    // copying data..
    void * data = static_cast<uint8_t*>(exportMemory) + ldMemory->GetSize();
    model.CopyData(data, model.GetSize());

    return exportMemory;
}

void HardwareModelSue1::PopulateHeader(intel_gna_model_header & modelHeader) const
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
