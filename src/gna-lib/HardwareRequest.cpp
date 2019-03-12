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

#include "HardwareRequest.h"

#include "HardwareLayer.h"
#include "HardwareModelScorable.h"
#include "LayerConfiguration.h"
#include "RequestConfiguration.h"

using namespace GNA;

HardwareRequest::HardwareRequest(const HardwareModelScorable& hwModelIn,
                                const RequestConfiguration& requestConfigurationIn,
                                Memory *ldMemoryIn, const std::vector<Memory *>& modelMemoryObjects)
    : HwPerfEncoding(requestConfigurationIn.HwPerfEncoding),
      RequestConfigId(requestConfigurationIn.Id),
      requestConfiguration(requestConfigurationIn),
      hwModel(hwModelIn),
      ldMemory(ldMemoryIn)
{
    DriverMemoryObjects.push_back(DriverBuffer{ ldMemory });
    for (auto memory : modelMemoryObjects)
    {
        DriverMemoryObjects.push_back(DriverBuffer{ memory });
    }
    for (auto memory : requestConfiguration.MemoryList)
    {
        DriverMemoryObjects.push_back(DriverBuffer{ memory });
    }
    Invalidate();
}

void HardwareRequest::Invalidate()
{
    auto& ldPatches = DriverMemoryObjects.front().Patches;
    ldPatches.clear();

    auto& model = requestConfiguration.Model;
    auto& layerConfigurations = requestConfiguration.LayerConfigurations;

    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        auto layer = model.GetLayer(it->first);
        auto hwLayer = hwModel.GetLayer(it->first);
        auto layerCfg = it->second.get();

        generateBufferPatches(*layerCfg, *layer, *hwLayer);

        if (layerCfg->ActList)
        {
            if (INTEL_GMM != layer->Operation)
            {
                auto activeList = it->second->ActList.get();

                auto ldActlistOffset = hwLayer->GetLdActlistOffset();
                auto actlistOffset = hwModel.GetBufferOffsetForConfiguration(
                                    activeList->Indices, requestConfiguration);

                auto ldActlenOffset = hwLayer->GetLdActlenOffset();
                uint16_t indices = static_cast<uint16_t>(activeList->IndicesCount);

                ldPatches.push_back({ ldActlistOffset, actlistOffset, sizeof(uint32_t) });
                ldPatches.push_back({ ldActlenOffset, indices, sizeof(uint32_t) });
            }
            else
            {
                auto activeList = it->second->ActList.get();

                auto ldActlistOffset = hwLayer->GetLdActlistOffset();
                auto asladdr = hwModel.GetBufferOffsetForConfiguration(
                                    activeList->Indices, requestConfiguration);

                auto ldActlenOffset = hwLayer->GetLdActlenOffset();
                auto indices = activeList->IndicesCount;

                auto ldScrlenOffset = hwLayer->GetLdScrlenOffset();
                auto scrlen = hwLayer->GetScrlen(activeList->IndicesCount);

                ldPatches.push_back({ldActlistOffset, asladdr, sizeof(ASLADDR)});
                ldPatches.push_back({ldActlenOffset, indices, sizeof(ASTLISTLEN)});
                ldPatches.push_back({ldScrlenOffset, scrlen, sizeof(GMMSCRLEN)});
            }
        }
    }
}

void HardwareRequest::Update(uint32_t layerIndex, uint32_t layerCount, GnaOperationMode mode)
{
    auto hwLayer = hwModel.GetLayer(layerIndex);

    Mode = mode;
    LayerBase = hwLayer->GetXnnDescriptorOffset();
    if(GMM == mode)
    {
        GmmOffset = hwLayer->GetGmmDescriptorOffset();
    }
    LayerCount = layerCount;

    updateActiveLists(layerIndex, layerCount);
    ActiveListOn = activeLists.at(layerIndex);
}

void HardwareRequest::generateBufferPatches(const LayerConfiguration& layerConfiguration,
        const Layer &layer, const HardwareLayer &hwLayer)
{
    const auto& buffers = layerConfiguration.Buffers;
    auto& ldPatches = DriverMemoryObjects.front().Patches;

    for (auto it = buffers.cbegin(); it != buffers.cend(); it++)
    {
        auto componentType = it->first;
        auto address = it->second;

        uint32_t bufferOffset = hwModel.GetBufferOffsetForConfiguration(address, requestConfiguration);
        uint32_t ldOffset = 0;
        switch (componentType)
        {
            case InputComponent:
                ldOffset = hwLayer.GetLdInputOffset();
                break;
            case OutputComponent:
            {
                ldOffset = hwLayer.GetLdOutputOffset();
                if (INTEL_RECURRENT == layer.Operation)
                {
                    auto feedbackBufferOffset = hwModel.GetBufferOffsetForConfiguration(
                        layerConfiguration.Configs.Recurrent->feedbackBuffer, requestConfiguration);
                    auto ldFeedbackOffset = hwLayer.GetLdFeedbackOffset();
                    ldPatches.push_back({ ldFeedbackOffset, feedbackBufferOffset, sizeof(uint32_t) });
                }
                else if (layer.Operation == INTEL_AFFINE || layer.Operation == INTEL_GMM)
                {
                    auto nnopTypeOffset = hwLayer.GetLdNnopOffset();
                    auto nnopTypeValue = hwLayer.GetNnopType(layerConfiguration.ActList.get() != nullptr);
                    ldPatches.push_back({ nnopTypeOffset, nnopTypeValue, sizeof(uint8_t) });
                }
                break;
            }
            case IntermediateOutputComponent:
                ldOffset = hwLayer.GetLdIntermediateOutputOffset();
                break;
            // TODO:3: support updating below components
            //case WeightComponent:
            //    ldOffset = hwLayer.GetLdWeightOffset();
            //    break;
            //case BiasComponent:
            //    ldOffset = hwLayer.GetLdBiasOffset();
            //    break;
            //case WeightScaleFactorComponent:
            //    ldOffset = hwLayer.GetLdWeightScaleFactorOffset();
            //    break;
            //case PwlComponent:
            //    ldOffset = hwLayer.GetLdPwlOffset();
            //    break;
            //case GmmMeanComponent:
            //    ldOffset = hwLayer.GetLdGmmMeanOffset();
            //    break;
            //case GmmInverseCovarianceComponent:
            //    ldOffset = hwLayer.GetLdGmmInverseCovarianceOffset();
            //    break;
            //case GmmGaussianConstantComponent:
            //    ldOffset = hwLayer.GetLdGaussianConstantOffset();
            //    break;
            //case RecurrentComponent:
                //ldOffset = hwLayer.GetLdFeedbackOffset();
                //break;
            default:
                throw GnaException { GNA_UNKNOWN_ERROR };
        }

        ldPatches.push_back({ ldOffset, bufferOffset, sizeof(uint32_t) });
    }
}

void HardwareRequest::updateActiveLists(uint32_t layerIndex, uint32_t layerCount)
{
    auto& layerConfigurations = requestConfiguration.LayerConfigurations;
    auto& model = requestConfiguration.Model;
    if (activeLists.find(layerIndex) == activeLists.end())
    {
        activeLists[layerIndex] = false;
        auto lowerBound = layerConfigurations.lower_bound(layerIndex);
        auto upperBound = layerConfigurations.upper_bound(layerIndex + layerCount);
        for (auto it = lowerBound; it != upperBound; ++it)
        {
            auto layer = model.GetLayer(it->first);
            if (it->second->ActList && INTEL_GMM == layer->Operation)
            {
                activeLists[layerIndex] = true;
                break;
            }
        }
    }
}

