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
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

using namespace GNA;

size_t HardwareModel::CalculateDescriptorSize(const uint32_t layerCount, const uint16_t gmmLayersCount)
{
    Expect::InRange<uint32_t>(layerCount + gmmLayersCount, 1, XNN_LAYERS_MAX_COUNT + GMM_LAYERS_MAX_COUNT,
        XNN_ERR_NET_LYR_NO);
    auto layerDescriptorsSizeTmp = getLayerDescriptorsSize(layerCount);
    auto gmmDescriptorsSizeTmp = getGmmDescriptorsSize(gmmLayersCount);

    return layerDescriptorsSizeTmp + gmmDescriptorsSizeTmp;
}

HardwareModel::HardwareModel(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers,
    uint32_t gmmCount, const uint64_t memoryIdIn, const BaseAddress memoryBaseIn,
    const BaseAddress baseDescriptorAddress, IoctlSender &sender, const AccelerationDetector& detector) :
    memoryId{ memoryIdIn },
    memoryBase{ memoryBaseIn },
    modelId{ modId },
    baseDescriptor{ memoryBaseIn, baseDescriptorAddress, detector },
    ioctlSender{sender},
    softwareLayers{ layers },
    gmmDescriptorsSize{ getGmmDescriptorsSize(gmmCount) }
{
}

void HardwareModel::InvalidateConfig(gna_request_cfg_id configId)
{
    if(hardwareRequests.find(configId) != hardwareRequests.end())
        hardwareRequests[configId]->Invalidate();
}

status_t HardwareModel::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *buffers,
    const GnaOperationMode operationMode)
{
    UNREFERENCED_PARAMETER(buffers);

    auto configId = requestConfiguration.ConfigId;
    HardwareRequest *hwRequest = nullptr;

    if (hardwareRequests.find(configId) == hardwareRequests.end())
    {
        auto inserted = hardwareRequests.emplace(
            configId,
            std::make_unique<HardwareRequest>(memoryId, *this, requestConfiguration));
        hwRequest = inserted.first->second.get();
    }
    else
    {
        hwRequest = hardwareRequests.at(configId).get();
    }

    hwRequest->Update(layerIndex, layerCount, operationMode);

    auto result = ioctlSender.Submit(hwRequest, profiler);

    auto perfResults = requestConfiguration.PerfResults;
    if (perfResults)
    {
        perfResults->drv.startHW += result.driverPerf.startHW;
        perfResults->drv.scoreHW += result.driverPerf.scoreHW;
        perfResults->drv.intProc += result.driverPerf.intProc;

        perfResults->hw.stall += result.hardwarePerf.stall;
        perfResults->hw.total += result.hardwarePerf.total;
    }

    return result.status;
}

void HardwareModel::Build()
{
    auto gmmDescriptor = AddrGmmCfg();
    if (0 != gmmDescriptorsSize)
    {
        gmmDescriptor = AddrGmmCfg(baseDescriptor.GetMemAddress()) + static_cast<uint32_t>(softwareLayers.size());
    }
    auto layerDescriptor = LayerDescriptor(baseDescriptor, gmmDescriptor);
    auto i = uint32_t { 0 };
    for (auto& layer : softwareLayers)
    {
        try
        {
            const auto parameters = DescriptorParameters{layer.get(), layerDescriptor};
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
            throw GnaModelException(GnaException(XNN_ERR_LYR_CFG), i);
        }
    }
}

uint32_t HardwareModel::getLayerDescriptorsSize(const uint32_t layerCount)
{
    Expect::InRange<uint32_t>(layerCount, 0, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto layerDescriptorsSizeTmp = LayerDescriptor::GetSize(layerCount, GNA_CNL);
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(const uint32_t gmmLayersCount)
{
    Expect::InRange(gmmLayersCount, GMM_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto gmmDescriptorsSizeTmp = size_t{gmmLayersCount * sizeof(GMM_CONFIG)};
    return static_cast<uint32_t>(gmmDescriptorsSizeTmp);
}

