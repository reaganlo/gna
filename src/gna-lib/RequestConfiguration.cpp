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

#include "RequestConfiguration.h"

#include <memory>

#include "GnaException.h"
#include "LayerConfiguration.h"
#include "Validator.h"
#include "GnaDrvApi.h"
#include "GnaConfig.h"
#include "CompiledModel.h"

using namespace GNA;

RequestConfiguration::RequestConfiguration(CompiledModel& model, gna_request_cfg_id configId) :
    Model{model},
    ConfigId{configId}
{ }

void RequestConfiguration::AddBuffer(gna_buffer_type type, uint32_t layerIndex, void *address)
{
    const auto& layer = *Model.GetLayers().at(layerIndex);
    auto layerType = layer.Config.Type;
    if (INTEL_HIDDEN == layerType 
        || (GNA_IN == type && INTEL_OUTPUT == layerType) 
        || (GNA_OUT == type && INTEL_INPUT == layerType))
    {
        throw GnaException{ XNN_ERR_LYR_TYPE };
    }

    invalidateHwConfigCache();

    auto found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();
    switch (type)
    {
    case GNA_IN:
        Expect::Null(layerConfiguration->InputBuffer.get());
        layerConfiguration->InputBuffer = std::make_unique<ConfigurationBuffer>(GNA_IN, address);
        ++InputBuffersCount;
        break;
    case GNA_OUT:
        Expect::Null(layerConfiguration->OutputBuffer.get());
        layerConfiguration->OutputBuffer = std::make_unique<ConfigurationBuffer>(GNA_OUT, address);
        ++OutputBuffersCount;
        break;
    default:
        throw GnaException(GNA_UNKNOWN_ERROR);
    }

    layer.UpdateKernelConfigs(*layerConfiguration);
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, const ActiveList& activeList)
{
    const auto& layer = *Model.GetLayers().at(layerIndex);
    auto layerType = layer.Config.Type;
    if (INTEL_OUTPUT != layerType && INTEL_INPUT_OUTPUT != layerType)
    {
        throw GnaException{ XNN_ERR_LYR_TYPE };
    }
    auto layerKind = layer.Config.Kind;
    if (INTEL_AFFINE != layerKind && INTEL_GMM != layerKind)
    {
        throw GnaException{ XNN_ERR_LYR_KIND };
    }

    invalidateHwConfigCache();

    auto found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();
    Expect::Null(layerConfiguration->ActiveList.get());

    auto activeListPtr = ActiveList::Create(activeList, layer.Config.Kind);
    layerConfiguration->ActiveList.swap(activeListPtr);
    ++ActiveListCount;

    layer.UpdateKernelConfigs(*layerConfiguration);
}

// TODO: below methods are hardware related only thus should be somewhere closer to hardware classes
// then deep delegating calls chain will disappear
void RequestConfiguration::GetHwConfigData(void* &buffer, size_t &size, uint32_t layerIndex, uint32_t layerCount) const
{
    auto submodelConfigCache = hwConfigCaches[layerIndex].get();

    if (!submodelConfigCache)
    {
        calculateCacheSize(layerIndex, layerCount);

        hwConfigCaches[layerIndex].reset(new uint8_t[hwConfigSizes[layerIndex]]);
        submodelConfigCache = hwConfigCaches[layerIndex].get();

        auto calculationData = reinterpret_cast<PGNA_CALC_IN>(submodelConfigCache);
        calculationData->ctrlFlags.activeListOn = ActiveListCount > 0;
        calculationData->ctrlFlags.gnaMode = 1; // xnn by default
        calculationData->ctrlFlags.layerIndex = layerIndex;
        calculationData->ctrlFlags.layerCount = layerCount;
        calculationData->modelId = Model.Id;
        calculationData->hwPerfEncoding = HwPerfEncoding;
        calculationData->reqCfgDescr.buffersCount = InputBuffersCount + OutputBuffersCount;
        calculationData->reqCfgDescr.requestConfigId = ConfigId;

        void* bufferShifted = submodelConfigCache + sizeof(GNA_CALC_IN);
        writeBuffersIntoCache(layerIndex, layerCount, bufferShifted);
        writeNnopTypesIntoCache(layerIndex, layerCount, bufferShifted, calculationData->reqCfgDescr.nnopTypesCount);
        writeXnnActiveListsIntoCache(layerIndex, layerCount, bufferShifted, calculationData->reqCfgDescr.xnnActiveListsCount);
        writeGmmActiveListsIntoCache(layerIndex, layerCount, bufferShifted, calculationData->reqCfgDescr.gmmActiveListsCount);
    }

    buffer = submodelConfigCache;
    size = hwConfigSizes[layerIndex];
}

void RequestConfiguration::invalidateHwConfigCache()
{
    for (auto& it : hwConfigCaches)
    {
        it.second.reset();
    }
    for (auto& it : hwConfigSizes)
    {
        it.second = 0;
    }
}

void RequestConfiguration::calculateCacheSize(uint32_t layerIndex, uint32_t layerCount) const
{
    auto& hwConfigSize = hwConfigSizes[layerIndex];
    hwConfigSize = 0;

    if (!hwConfigCaches[layerIndex].get())
    {
        hwConfigSize = sizeof(GNA_CALC_IN);
        hwConfigSize += InputBuffersCount * sizeof(GNA_BUFFER_DESCR);
        hwConfigSize += OutputBuffersCount * sizeof(GNA_BUFFER_DESCR);

        // it's possible that AFFINE and GMM layers will need nnop change
        auto nnopLayersCount = 0ui32;
        const auto& layers = Model.GetLayers();
        auto lowerBound = LayerConfigurations.lower_bound(layerIndex);
        auto upperBound = LayerConfigurations.upper_bound(layerIndex + layerCount);
        for (auto it = lowerBound; it != upperBound; ++it)
        {
            auto layer = layers.at(it->first).get();
            if(it->second->OutputBuffer
                && (layer->Config.Kind == INTEL_AFFINE || layer->Config.Kind == INTEL_GMM))
            {
                ++nnopLayersCount;
            }
        }
        hwConfigSize += nnopLayersCount * sizeof(NNOP_TYPE_DESCR);

        // TODO: different counts shell be used, instead buffer might be bigger than needed, but still safe
        hwConfigSize += ActiveListCount * max(sizeof(XNN_ACTIVE_LIST_DESCR), sizeof(GMM_ACTIVE_LIST_DESCR));
    }
}

void RequestConfiguration::writeBuffersIntoCache(uint32_t layerIndex, uint32_t layerCount, void* &buffer) const
{
    auto lyrsCfg = reinterpret_cast<PGNA_BUFFER_DESCR>(buffer);

    auto lowerBound = LayerConfigurations.lower_bound(layerIndex);
    auto upperBound = LayerConfigurations.upper_bound(layerIndex + layerCount);
    for (auto it = lowerBound; it != upperBound; ++it)
    {
        if (it->second->InputBuffer)
        {
            Model.WriteHardwareLayerInputBuffer(it->first, lyrsCfg, it->second->InputBuffer.get());
            ++lyrsCfg;
        }

        if (it->second->OutputBuffer)
        {
            Model.WriteHardwareLayerOutputBuffer(it->first, lyrsCfg, it->second->OutputBuffer.get());
            ++lyrsCfg;
        }
    }

    buffer = lyrsCfg;
}

void RequestConfiguration::writeNnopTypesIntoCache(uint32_t layerIndex, uint32_t layerCount,
    void* &buffer, UINT32 &count) const
{
    auto nnopCfg = reinterpret_cast<PNNOP_TYPE_DESCR>(buffer);
    count = 0;

    const auto& layers = Model.GetLayers();

    auto lowerBound = LayerConfigurations.lower_bound(layerIndex);
    auto upperBound = LayerConfigurations.upper_bound(layerIndex + layerCount);
    for (auto it = lowerBound; it != upperBound; ++it)
    {
        auto layer = layers.at(it->first).get();
        if(it->second->OutputBuffer
            && (layer->Config.Kind == INTEL_AFFINE || layer->Config.Kind == INTEL_GMM))
        {
            Model.WriteHardwareLayerNnopType(it->first, nnopCfg, nullptr != it->second->ActiveList);

            ++nnopCfg;
            ++count;
        }
    }

    buffer = nnopCfg;
}

void RequestConfiguration::writeXnnActiveListsIntoCache(uint32_t layerIndex, uint32_t layerCount,
    void* &buffer, UINT32 &count) const
{
    auto actLstCfg = reinterpret_cast<PXNN_ACTIVE_LIST_DESCR>(buffer);
    count = 0;

    const auto& layers = Model.GetLayers();

    auto lowerBound = LayerConfigurations.lower_bound(layerIndex);
    auto upperBound = LayerConfigurations.upper_bound(layerIndex + layerCount);
    for (auto it = lowerBound; it != upperBound; ++it)
    {
        
        if (INTEL_GMM != layers[it->first]->Config.Kind && it->second->ActiveList)
        {
            HardwareActiveListDescriptor descriptor{it->second->ActiveList.get(), actLstCfg};
            Model.WriteHardwareLayerActiveList(it->first, descriptor);
            ++actLstCfg;
            ++count;
        }
    }

    buffer = actLstCfg;
}

void RequestConfiguration::writeGmmActiveListsIntoCache(uint32_t layerIndex, uint32_t layerCount,
    void* &buffer, UINT32 &count) const
{
    auto actLstCfg = reinterpret_cast<PGMM_ACTIVE_LIST_DESCR>(buffer);
    count = 0;

    const auto& layers = Model.GetLayers();

    auto lowerBound = LayerConfigurations.lower_bound(layerIndex);
    auto upperBound = LayerConfigurations.upper_bound(layerIndex + layerCount);
    for (auto it = lowerBound; it != upperBound; ++it)
    {
        if (INTEL_GMM == layers[it->first]->Config.Kind && it->second->ActiveList)
        {
            HardwareActiveListDescriptor descriptor{it->second->ActiveList.get(), actLstCfg};
            Model.WriteHardwareLayerActiveList(it->first, descriptor);
            ++actLstCfg;
            ++count;
        }
    }

    buffer = actLstCfg;
}

