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

#include "GnaException.h"
#include "HardwareLayer.h"
#include "LayerDescriptor.h"
#include "Layer.h"
#include "Memory.h"

#include "gna-api-status.h"

#include <algorithm>

using namespace GNA;


HardwareCapabilities HardwareModelSue1::sueCapabilities = HardwareCapabilities{ Gna2DeviceVersionSueCreek };

HardwareModelSue1::HardwareModelSue1(
    const std::vector<std::unique_ptr<Layer>>& layers, uint32_t gmmCount,
    std::unique_ptr<Memory> dumpMemory) :
    HardwareModel(layers, gmmCount, sueCapabilities)
{
    ldMemory = std::move(dumpMemory);
}

const LayerDescriptor& HardwareModelSue1::GetDescriptor(uint32_t layerIndex) const
{
    return hardwareLayers.at(layerIndex)->XnnDescriptor;
}

uint32_t HardwareModelSue1::GetOutputOffset(uint32_t layerIndex) const
{
    auto layer = hardwareLayers.at(layerIndex).get();
    return layer->GetLdOutputOffset() - GetDescriptor(0).GetOffset();
}

uint32_t HardwareModelSue1::GetInputOffset(uint32_t layerIndex) const
{
    auto layer = hardwareLayers.at(layerIndex).get();
    return layer->GetLdInputOffset () - GetDescriptor(0).GetOffset();
}

void HardwareModelSue1::allocateLayerDescriptors()
{
    baseDescriptor = std::make_unique<LayerDescriptor>(
            *ldMemory, ldMemory->GetBuffer(), hwCapabilities);
    if (!baseDescriptor)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }
}

uint32_t HardwareModelSue1::GetBufferOffset(const BaseAddress& address) const
{
    if (address.InRange(ldMemory->GetBuffer(),
        static_cast<uint32_t>(ldMemory->GetSize())))
    {
        return address.GetOffset(BaseAddress{ ldMemory->GetBuffer() });
    }

    auto offset = static_cast<uint32_t>(ldMemory->GetSize());
    for (auto memory : modelMemoryObjects)
    {
        if (address.InRange(memory->GetBuffer(),
            static_cast<uint32_t>(memory->GetSize())))
        {
            return offset + address.GetOffset(BaseAddress{memory->GetBuffer()});
        }

        offset += static_cast<uint32_t>(memory->GetSize());
    }

    return 0;
}
