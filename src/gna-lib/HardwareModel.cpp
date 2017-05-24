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
#include "HardwareLayer.h"
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

using namespace GNA;

const size_t HardwareModel::CalculateDescriptorSize(const uint16_t layerCount, const uint16_t gmmLayersCount)
{
    auto layerDescriptorsSizeTmp = getLayerDescriptorsSize(layerCount);
    auto gmmDescriptorsSizeTmp = getGmmDescriptorsSize(gmmLayersCount);

    return layerDescriptorsSizeTmp + gmmDescriptorsSizeTmp;
}

HardwareModel::HardwareModel(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers,
    const Memory& wholeMemory, const AccelerationDetector& detector) :
    modelId{modId},
    memoryBaseAddress{wholeMemory},
    layerDescriptorsSize{getLayerDescriptorsSize(XNN_LAYERS_MAX_COUNT)}, // TODO: change to support variable number of layers
    gmmDescriptorsSize{getGmmDescriptorsSize(XNN_LAYERS_MAX_COUNT)} // TODO: change to support variable number of gmms
{
    build(layers, detector.GetHardwareBufferSize());
}

status_t HardwareModel::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    UNREFERENCED_PARAMETER(buffers);

    void* data;
    size_t size;
    requestConfiguration.GetHwConfigData(data, size, layerIndex, layerCount);

    sender.Submit(data, size, profiler);

    auto response = reinterpret_cast<PGNA_CALC_IN>(data);
    auto status = response->status;
    Expect::True(GNA_SUCCESS == status || GNA_SSATURATE == status, status);

    return status;
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
        const auto parameters = DescriptorParameters{layer.get(), memoryBaseAddress, layerDescriptor, gmmDescriptor,
            hardwareInternalBufferSize};
        hardwareLayers.push_back(HardwareLayer::Create(parameters));
        layerDescriptor++;
        if (INTEL_GMM == layer->Config.Kind)
        {
            gmmDescriptor++;
        }
    }
}

uint32_t HardwareModel::getLayerDescriptorsSize(const uint16_t layerCount)
{
    Expect::InRange(layerCount, 1, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto layerDescriptorsSizeTmp = size_t{layerCount * sizeof(XNN_LYR)};
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(const uint16_t gmmLayersCount)
{
    Expect::InRange(gmmLayersCount, 0, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto gmmDescriptorsSizeTmp = size_t{gmmLayersCount * sizeof(GMM_CONFIG)};
    return gmmDescriptorsSizeTmp;
}

