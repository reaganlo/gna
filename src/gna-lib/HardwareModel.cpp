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

#include "HardwareModel.h"

#include "AccelerationDetector.h"
#include "GnaDrvApi.h"
#include "GnaException.h"
#include "HardwareLayer.h"
#include "Validator.h"

using namespace GNA;

const size_t HardwareModel::CalculateDescriptorSize(const uint16_t layerCount, const uint16_t gmmLayersCount)
{
    auto layerDescriptorsSize = getLayerDescriptorsSize(layerCount);
    auto gmmDescriptorsSize = getGmmDescriptorsSize(gmmLayersCount);

    return layerDescriptorsSize + gmmDescriptorsSize;
}

HardwareModel::HardwareModel(const gna_model_id modId, const SoftwareModel& model, const Memory& wholeMemory,
    const AccelerationDetector& detector) :
    modelId(modId),
    memoryBaseAddress(wholeMemory),
    layerDescriptorsSize(getLayerDescriptorsSize(XNN_LAYERS_MAX_COUNT)), // TODO: change to support variable number of layers
    gmmDescriptorsSize(getGmmDescriptorsSize(XNN_LAYERS_MAX_COUNT)) // TODO: change to support variable number of gmms
{
    mapMemory(wholeMemory);//TODO:INTEGRATION: move to higher level and map after models are compiled
    build(model.Layers, detector.GetHardwareBufferSize());
}

HardwareModel::~HardwareModel()
{
    unmapMemory();
}

void HardwareModel::mapMemory(const Memory& memory)
{
    if (memoryMapped)
        throw GnaException(GNA_UNKNOWN_ERROR);

    // write model id in user buffer
    // driver will retrieve it
    *reinterpret_cast<uint64_t*>(memory.Get()) = static_cast<uint64_t>(modelId);

    IoctlSend(
        GNA_IOCTL_MEM_MAP,
        nullptr,
        0,
        memory.Get(),
        memory.GetSize());

    memoryMapped = true;
}

void HardwareModel::unmapMemory()
{
    uint64_t mId = static_cast<uint64_t>(modelId);
    IoctlSend(GNA_IOCTL_MEM_UNMAP, &mId, sizeof(mId), nullptr, 0);

    memoryMapped = false;
}

void HardwareModel::build(const std::vector<std::unique_ptr<Layer>>& layers, const uint32_t hardwareInternalBufferSize)
{
    auto layerDescriptor = AddrXnnLyr(memoryBaseAddress);
    auto gmmDescriptor = AddrGmmCfg();
    if (0 != gmmDescriptorsSize)
    {
        gmmDescriptor = AddrGmmCfg(layerDescriptor.Get<uint8_t>() + layerDescriptorsSize);
    }

    for (auto& layer : layers)
    {
        *layerDescriptor = HardwareLayer::Convert(*layer, memoryBaseAddress, gmmDescriptor, hardwareInternalBufferSize);
        layerDescriptor++;
        if (INTEL_GMM == layer->Config.Kind)
        {
            gmmDescriptor++;
        }
    }
}