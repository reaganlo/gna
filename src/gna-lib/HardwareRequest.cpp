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

#include "HardwareModel.h"
#include "LayerConfiguration.h"
#include "RequestConfiguration.h"

using namespace GNA;

HardwareRequest::HardwareRequest(uint64_t memoryId, const HardwareModel& hwModelIn,
                                const RequestConfiguration& requestConfigurationIn)
    : MemoryId(memoryId), ModelId(requestConfigurationIn.Model.Id),
      HwPerfEncoding(requestConfigurationIn.HwPerfEncoding),
      RequestConfigId(requestConfigurationIn.ConfigId),
      requestConfiguration(requestConfigurationIn), hwModel(hwModelIn)
{
    Invalidate();
}

void HardwareRequest::Invalidate()
{
    IoBuffers.clear();
    NnopTypes.clear();
    XnnActiveLists.clear();
    GmmActiveLists.clear();

    auto& model = requestConfiguration.Model;
    auto& layerConfigurations = requestConfiguration.LayerConfigurations;
    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        auto layer = model.GetLayer(it->first);
        auto hwLayer = hwModel.GetLayer(it->first);
        auto layerCfg = it->second.get();

        if (layerCfg->InputBuffer)
        {
            auto bufferInputOffset = hwModel.GetOffset(*it->second->InputBuffer);
            auto ldInputOffset = hwLayer->GetLdInputOffset();
            IoBuffers.emplace_back(IoBufferPatch{ldInputOffset, bufferInputOffset});
        }

        if (layerCfg->OutputBuffer)
        {
            auto bufferOutputOffset = hwModel.GetOffset(*it->second->OutputBuffer);
            auto ldOutputOffset = hwLayer->GetLdOutputOffset();
            IoBuffers.emplace_back(IoBufferPatch{ldOutputOffset, bufferOutputOffset});

            if (layer->Config.Kind == INTEL_RECURRENT)
            {
                auto rnnLayer = dynamic_cast<const HardwareLayerRnn*>(hwLayer);
                auto bufferFeedbackOffset = rnnLayer->CalculateFeedbackBuffer(*layerCfg->OutputBuffer);
                auto ldFeedbackOffset = hwLayer->GetLdFeedbackOffset();
                IoBuffers.emplace_back(IoBufferPatch{ldFeedbackOffset, bufferFeedbackOffset});
            }
        }

        if (layerCfg->ActList)
        {
            if (INTEL_GMM != layer->Config.Kind)
            {
                auto ldActlistOffset = hwLayer->GetLdActlistOffset();
                auto ldActlenOffset = hwLayer->GetLdActlenOffset();
                auto activeList = it->second->ActList.get();
                auto actlistOffset = hwModel.GetOffset(activeList->Indices);
                uint16_t indices = activeList->IndicesCount;

                XnnActiveLists.emplace_back(XnnAlPatch{ldActlistOffset, actlistOffset,
                                                        ldActlenOffset, indices});
            }
            else
            {
                auto ldActlistOffset = hwLayer->GetLdActlistOffset();
                auto ldActlenOffset = hwLayer->GetLdActlenOffset();
                auto ldScrlenOffset = hwLayer->GetLdScrlenOffset();

                auto activeList = it->second->ActList.get();
                auto scrlen = hwLayer->GetScrlen(activeList->IndicesCount);
                auto asladdr = hwModel.GetOffset(activeList->Indices);
                auto indices = activeList->IndicesCount;

                GmmActiveLists.emplace_back(GmmAlPatch{ldActlistOffset, asladdr,
                                                        ldActlenOffset, indices,
                                                        ldScrlenOffset, scrlen});
            }
        }

        if (layerCfg->OutputBuffer &&
            (layer->Config.Kind == INTEL_AFFINE || layer->Config.Kind == INTEL_GMM))
        {
            auto nnopTypeOffset = hwLayer->GetLdNnopOffset();
            auto nnopTypeValue = hwLayer->GetNnopType(layerCfg->ActList.get() != nullptr);

            NnopTypes.emplace_back(NnopTypePatch{ nnopTypeOffset, nnopTypeValue });
        }
    }
}

void HardwareRequest::Update(uint32_t layerIndex, uint32_t layerCount, GnaOperationMode mode)
{
    auto hwLayer = hwModel.GetLayer(layerIndex);

    Mode = mode;
    LayerBase = hwLayer->GetLayerDescriptorOffset();
    if(GMM == mode)
    {
        GmmOffset = hwLayer->GetGmmDescriptorOffset();
    }
    LayerCount = layerCount;

    updateActiveLists(layerIndex, layerCount);
    ActiveListOn = activeLists.at(layerIndex);
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
            if (it->second->ActList && INTEL_GMM == layer->Config.Kind)
            {
                activeLists[layerIndex] = true;
                break;
            }
        }
    }
}

