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

using namespace GNA;

size_t HardwareModelSue1::CalculateDescriptorSize(const uint32_t layerCount, const uint16_t gmmLayersCount)
{
    Expect::Zero(gmmLayersCount, XNN_ERR_NET_LYR_NO);
    Expect::InRange<uint32_t>(layerCount, 1, XNN_LAYERS_MAX_COUNT + GMM_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    return getLayerDescriptorsSize(layerCount);
}

HardwareModelSue1::HardwareModelSue1(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers,
    const Memory &memoryIn, const BaseAddress &dumpDescriptorAddr, IoctlSender &sender, const AccelerationDetector& detector) :
    HardwareModel(
        modId,
        layers,
        0,
        0,
        memoryIn.Get() + memoryIn.InternalSize - getLayerDescriptorsSize(static_cast<uint32_t>(layers.size())),
        memoryIn.GetDescriptorsBase(modId),
        sender,
        detector)
{
    UNREFERENCED_PARAMETER(dumpDescriptorAddr);
}

uint32_t HardwareModelSue1::getLayerDescriptorsSize(const uint32_t layerCount)
{
    Expect::InRange<uint32_t>(layerCount, 0, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto layerDescriptorsSizeTmp =
        ALIGN(LayerDescriptor::GetSize(layerCount, GNA_SUE_CREEK), 0x1000);
    return layerDescriptorsSizeTmp;
}

const LayerDescriptor& HardwareModelSue1::GetDescriptor(uint32_t layerIndex) const
{
    return hardwareLayers.at(layerIndex)->XnnDescriptor;
};

uint32_t HardwareModelSue1::GetOutputOffset(uint32_t layerIndex) const
{
    auto layer = hardwareLayers.at(layerIndex).get();
    return layer->GetLdOutputOffset();
};

uint32_t HardwareModelSue1::GetInputOffset(uint32_t layerIndex) const
{
    auto layer = hardwareLayers.at(layerIndex).get();
    return layer->XnnDescriptor[in_buffer].GetOffset();

};
