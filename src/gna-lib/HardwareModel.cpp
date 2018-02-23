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

#include "AccelerationDetector.h"
#include "HardwareLayer.h"
#include "LayerConfiguration.h"
#include "Memory.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "SoftwareModel.h"

using namespace GNA;

const size_t HardwareModel::CalculateDescriptorSize(const uint16_t layerCount, const uint16_t gmmLayersCount)
{
    Expect::InRange(layerCount + gmmLayersCount, 1, XNN_LAYERS_MAX_COUNT + GMM_LAYERS_MAX_COUNT,
        XNN_ERR_NET_LYR_NO);
    auto layerDescriptorsSizeTmp = getLayerDescriptorsSize(layerCount);
    auto gmmDescriptorsSizeTmp = getGmmDescriptorsSize(gmmLayersCount);

    return layerDescriptorsSizeTmp + gmmDescriptorsSizeTmp;
}

HardwareModel::HardwareModel(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers,
    uint16_t gmmCount, const uint64_t memoryIdIn, const BaseAddressC memoryBaseIn, const BaseAddressC descriptorBaseIn, IoctlSender &sender, const AccelerationDetector& detector) :
    memoryId{ memoryIdIn },
    memoryBase{ memoryBaseIn },
    modelId{modId},
    descriptorsAddress{descriptorBaseIn},
    layerDescriptorsSize{ getLayerDescriptorsSize(layers.size()) },
    hardwareBufferSize{ detector.GetHardwareBufferSize() },
    ioctlSender{sender},
    softwareLayers{ layers },
    gmmDescriptorsSize{ getGmmDescriptorsSize(gmmCount) }
{
}
void HardwareModel::InvalidateConfigCache(gna_request_cfg_id configId)
{
    requestHwCaches[configId].reset();
    requestCacheSizes[configId] = 0;
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

    void* data;
    size_t size;
    getHwConfigData(data, size, layerIndex, layerCount, requestConfiguration, operationMode);

    ioctlSender.Submit(data, size, profiler);

    auto response = reinterpret_cast<PGNA_CALC_IN>(data);
    auto status = response->status;

    auto perfResults = requestConfiguration.PerfResults;
    if (perfResults)
    {
        perfResults->drv.startHW += response->drvPerf.startHW;
        perfResults->drv.scoreHW += response->drvPerf.scoreHW;
        perfResults->drv.intProc += response->drvPerf.intProc;

        perfResults->hw.stall += response->hwPerf.stall;
        perfResults->hw.total += response->hwPerf.total;
    }

    return status;
}

void HardwareModel::Build()
{
    auto layerDescriptor = AddrXnnLyr(descriptorsAddress);
    auto gmmDescriptor = AddrGmmCfg();
    if (0 != gmmDescriptorsSize)
    {
        gmmDescriptor = AddrGmmCfg(layerDescriptor + softwareLayers.size());
    }
    auto i = 0ui32;
    for (auto& layer : softwareLayers)
    {
        try
        {
            const auto parameters = DescriptorParameters{layer.get(), memoryBase, layerDescriptor, gmmDescriptor,
                hardwareBufferSize};
            hardwareLayers.push_back(HardwareLayer::Create(parameters));
            layerDescriptor++;
            if (INTEL_GMM == layer->Config.Kind)
            {
                gmmDescriptor++;
            }
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

uint32_t HardwareModel::getLayerDescriptorsSize(const uint16_t layerCount)
{
    Expect::InRange(layerCount, 0, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto layerDescriptorsSizeTmp = size_t{layerCount * sizeof(XNN_LYR)};
    return layerDescriptorsSizeTmp;
}

uint32_t HardwareModel::getGmmDescriptorsSize(const uint16_t gmmLayersCount)
{
    Expect::InRange(gmmLayersCount, 0, GMM_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    auto gmmDescriptorsSizeTmp = size_t{gmmLayersCount * sizeof(GMM_CONFIG)};
    return gmmDescriptorsSizeTmp;
}

void HardwareModel::getHwConfigData(void* &buffer, size_t &size, uint16_t layerIndex, uint16_t layerCount,
    const RequestConfiguration& requestConfiguration, const GnaOperationMode operationMode) const
{
    const auto& layerConfigurations = requestConfiguration.LayerConfigurations;

    auto& requestActiveLists = activeLists[requestConfiguration.ConfigId];
    if (requestActiveLists.find(layerIndex) == requestActiveLists.end())
    {
        requestActiveLists[layerIndex] = false;
        auto lowerBound = layerConfigurations.lower_bound(layerIndex);
        auto upperBound = layerConfigurations.upper_bound(layerIndex + layerCount);
        for (auto it = lowerBound; it != upperBound; ++it)
        {
            auto layer = softwareLayers.at(it->first).get();
            if (it->second->ActiveList && INTEL_GMM == layer->Config.Kind)
            {
                requestActiveLists[layerIndex] = true;
                break;
            }
        }
    }

    auto& requestCache = requestHwCaches.at(requestConfiguration.ConfigId);
    if (!requestCache)
    {
        uint32_t buffersCount = 0;
        uint32_t activeListCount = 0;
        uint32_t xnnActiveListCount = 0;
        uint32_t gmmActiveListCount = 0;
        uint32_t nnopTypesCount = 0;

        for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
        {
            auto layer = softwareLayers.at(it->first).get();
            if (it->second->ActiveList)
            {
                (INTEL_AFFINE == layer->Config.Kind) ? ++xnnActiveListCount : ++gmmActiveListCount;
            }
            if (it->second->InputBuffer)
            {
                ++buffersCount;
            }
            if (it->second->OutputBuffer)
            {
                ++buffersCount;

                // for rnn fb buffer offset
                if (INTEL_RECURRENT == layer->Config.Kind)
                {
                    ++buffersCount;
                }
            }
            if (it->second->OutputBuffer && (INTEL_AFFINE == layer->Config.Kind || INTEL_GMM == layer->Config.Kind))
            {
                ++nnopTypesCount;
            }
        }

        activeListCount = xnnActiveListCount + gmmActiveListCount;
        requestCacheSizes[requestConfiguration.ConfigId] = calculateCacheSize(buffersCount, nnopTypesCount, activeListCount);

        requestCache.reset(new uint8_t[requestCacheSizes.at(requestConfiguration.ConfigId)]);

        auto calculationData = reinterpret_cast<PGNA_CALC_IN>(requestCache.get());
        calculationData->ctrlFlags.activeListOn = gmmActiveListCount > 0;
        calculationData->memoryId = memoryId;
        calculationData->modelId = modelId;
        calculationData->hwPerfEncoding = requestConfiguration.HwPerfEncoding;
        calculationData->reqCfgDescr.requestConfigId = requestConfiguration.ConfigId;
        calculationData->reqCfgDescr.buffersCount = buffersCount;
        calculationData->reqCfgDescr.xnnActiveListsCount = xnnActiveListCount;
        calculationData->reqCfgDescr.gmmActiveListsCount = gmmActiveListCount;
        calculationData->reqCfgDescr.nnopTypesCount = nnopTypesCount;

        void* bufferShifted = requestCache.get() + sizeof(GNA_CALC_IN);
        writeBuffersIntoCache(bufferShifted, requestConfiguration.LayerConfigurations);
        writeNnopTypesIntoCache(bufferShifted, requestConfiguration.LayerConfigurations);
        writeXnnActiveListsIntoCache(bufferShifted, requestConfiguration.LayerConfigurations);
        writeGmmActiveListsIntoCache(bufferShifted, requestConfiguration.LayerConfigurations);
    }

    auto calculationData = reinterpret_cast<PGNA_CALC_IN>(requestCache.get());
    calculationData->ctrlFlags.gnaMode = operationMode;
    if (xNN == operationMode)
    {
        calculationData->ctrlFlags.layerBase = GetOffset(descriptorsAddress) + layerIndex * sizeof(XNN_LYR);
    }
    else
    {
        calculationData->ctrlFlags.gmmOffset = hardwareLayers.at(layerIndex)->XnnDescriptor->gmm_descriptor;
    }
    calculationData->ctrlFlags.activeListOn = requestActiveLists.at(layerIndex);
    calculationData->ctrlFlags.layerCount = layerCount;

    buffer = requestCache.get();
    size = requestCacheSizes.at(requestConfiguration.ConfigId);
}

size_t HardwareModel::calculateCacheSize(uint32_t buffersCount, uint32_t nnopLayersCount, uint32_t activeListCount) const
{
    uint32_t cacheSize = sizeof(GNA_CALC_IN);
    cacheSize += buffersCount * sizeof(GNA_BUFFER_DESCR);
    cacheSize += nnopLayersCount * sizeof(NNOP_TYPE_DESCR);
    cacheSize += activeListCount * max(sizeof(XNN_ACTIVE_LIST_DESCR), sizeof(GMM_ACTIVE_LIST_DESCR));
    cacheSize = ALIGN(cacheSize, sizeof UINT64);

    return cacheSize;
}

void HardwareModel::writeBuffersIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const
{
    auto lyrsCfg = reinterpret_cast<PGNA_BUFFER_DESCR>(buffer);
    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        auto& hwLayer = hardwareLayers.at(it->first);
        if (it->second->InputBuffer)
        {
            hwLayer->WriteInputBuffer(lyrsCfg, it->second->InputBuffer.get());
        }

        if (it->second->OutputBuffer)
        {
            hwLayer->WriteOutputBuffer(lyrsCfg, it->second->OutputBuffer.get());
        }
    }

    buffer = lyrsCfg;
}

void HardwareModel::writeNnopTypesIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const
{
    auto nnopCfg = reinterpret_cast<PNNOP_TYPE_DESCR>(buffer);
    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        auto layer = softwareLayers.at(it->first).get();
        if(it->second->OutputBuffer
            && (layer->Config.Kind == INTEL_AFFINE || layer->Config.Kind == INTEL_GMM))
        {
            hardwareLayers.at(it->first)->WriteNnopType(nnopCfg, nullptr != it->second->ActiveList);
            ++nnopCfg;
        }
    }

    buffer = nnopCfg;
}

void HardwareModel::writeXnnActiveListsIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const
{
    auto actLstCfg = reinterpret_cast<PXNN_ACTIVE_LIST_DESCR>(buffer);

    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        if (INTEL_GMM != softwareLayers.at(it->first)->Config.Kind && it->second->ActiveList)
        {
            HardwareActiveListDescriptor descriptor{it->second->ActiveList.get(), actLstCfg};
            hardwareLayers.at(it->first)->WriteActiveList(descriptor);
            ++actLstCfg;
        }
    }

    buffer = actLstCfg;
}

void HardwareModel::writeGmmActiveListsIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const
{
    auto actLstCfg = reinterpret_cast<PGMM_ACTIVE_LIST_DESCR>(buffer);
    for (auto it = layerConfigurations.cbegin(); it != layerConfigurations.cend(); ++it)
    {
        if (INTEL_GMM == softwareLayers.at(it->first)->Config.Kind && it->second->ActiveList)
        {
            HardwareActiveListDescriptor descriptor{it->second->ActiveList.get(), actLstCfg};
            hardwareLayers.at(it->first)->WriteActiveList(descriptor);
            ++actLstCfg;
        }
    }

    buffer = actLstCfg;
}
