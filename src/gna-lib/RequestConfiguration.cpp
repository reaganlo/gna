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

#include <memory>

#include "GnaException.h"
#include "RequestConfiguration.h"
#include "Validator.h"
#include "GnaDrvApi.h"
#include "GnaConfig.h"
#include "CompiledModel.h"

using namespace GNA;

RequestConfiguration::RequestConfiguration(const CompiledModel& model, gna_request_cfg_id configId) :
    Model(model),
    ConfigId(configId)
{}

void RequestConfiguration::AddBuffer(gna_buffer_type type, uint32_t layerIndex, void *address)
{
    invalidateHwConfigCache();

    auto found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();
    // TODO:INTEGRATION add model validation - verify if RequestConfiguration is  valid for given layer
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
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, uint32_t indicesCount, uint32_t *indices)
{
    invalidateHwConfigCache();

    // TODO:INTEGRATION add model validation - verify if RequestConfiguration is  valid for given layer
    auto found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();
    Expect::Null(layerConfiguration->ActiveList.get());
    layerConfiguration->ActiveList = std::make_unique<ActiveList>(indicesCount, indices);
    ++ActiveListCount;
}

void GNA::RequestConfiguration::GetHwConfigData(void* &buffer, size_t &size) const
{
    if (!hwConfigCache)
    {
        auto bufCnfgCnt = InputBuffersCount + OutputBuffersCount;

        hwConfigSize = sizeof(GNA_CALC_IN);
        hwConfigSize += bufCnfgCnt * sizeof(GNA_BUFFER_DESCR);
        hwConfigSize += ActiveListCount * sizeof(GNA_ACTIVE_LIST_DESCR);
        hwConfigCache.reset(new uint8_t[hwConfigSize]);

        auto calculationData = reinterpret_cast<PGNA_CALC_IN>(hwConfigCache.get());

        calculationData->ctrlFlags.activeListOn = ActiveListCount > 0;
        calculationData->ctrlFlags.gnaMode = 1; // xnn by default
        calculationData->ctrlFlags.layerCount = Model.GetLayerCount();
        calculationData->ctrlFlags.layerIndex = 0;
        calculationData->modelId = Model.GetModelId();
        calculationData->ctrlFlags.bufferConfigsCount = bufCnfgCnt;
        calculationData->ctrlFlags.actListConfigsCount = ActiveListCount;
        calculationData->hwPerfEncoding = HwPerfEncoding;

        auto lyrsCfg = reinterpret_cast<PGNA_BUFFER_DESCR>(hwConfigCache.get() + sizeof(GNA_CALC_IN));
        writeLayerConfigBuffersIntoHwConfigCache(lyrsCfg);

        auto actLstCfg = reinterpret_cast<PGNA_ACTIVE_LIST_DESCR>(lyrsCfg);
        writeLayerConfigActiveListsIntoHwConfigCache(actLstCfg);
    }

    buffer = hwConfigCache.get();
    size = hwConfigSize;
}

void GNA::RequestConfiguration::invalidateHwConfigCache()
{
    hwConfigSize = 0;
    hwConfigCache.reset();
}

void GNA::RequestConfiguration::writeLayerConfigBuffersIntoHwConfigCache(PGNA_BUFFER_DESCR &lyrsCfg) const
{
    auto& hwModel = Model.GetHardwareModel();

    for (const auto& lc : LayerConfigurations)
    {
        if (lc.second->InputBuffer)
        {
            lyrsCfg->offset = lc.first * sizeof(XNN_LYR) + offsetof(XNN_LYR, in_buffer);
            lyrsCfg->value = hwModel.GetOffset(*lc.second->InputBuffer);
            ++lyrsCfg;
        }

        if (lc.second->OutputBuffer)
        {
            if (LayerOutput::ActivatedOutput == Model.GetSoftwareModel().Layers[lc.first]->Output.GetOutputMode())
            {
                lyrsCfg->offset = lc.first * sizeof(XNN_LYR) + offsetof(XNN_LYR, out_act_fn_buffer);
            }
            else
            {
                lyrsCfg->offset = lc.first * sizeof(XNN_LYR) + offsetof(XNN_LYR, out_sum_buffer);
            }
            lyrsCfg->value = hwModel.GetOffset(*lc.second->OutputBuffer);
            ++lyrsCfg;
        }
    }
}

void GNA::RequestConfiguration::writeLayerConfigActiveListsIntoHwConfigCache(PGNA_ACTIVE_LIST_DESCR &actLstCfg) const
{
    auto& hwModel = Model.GetHardwareModel();

    for (const auto& lc : LayerConfigurations)
    {
        if (lc.second->ActiveList)
        {
            // TODO: XNN_LYR.NN_OP_TYPE needs to be set to Active List type
            actLstCfg->act_list_buffer_offset = lc.first * sizeof(XNN_LYR) + offsetof(XNN_LYR, act_list_buffer);
            actLstCfg->act_list_buffer_value = hwModel.GetOffset((void*)lc.second->ActiveList->Indices);
            actLstCfg->act_list_n_elems_offset = lc.first * sizeof(XNN_LYR) + offsetof(XNN_LYR, act_list_n_elems);
            actLstCfg->act_list_buffer_value = lc.second->ActiveList->IndicesCount;
            ++actLstCfg;
        }
        // TODO: else: XNN_LYR.NN_OP_TYPE needs to be set to Non Active List type
    }
}

ConfigurationBuffer::ConfigurationBuffer(gna_buffer_type type, void* address) :
    InOutBuffer{address},
    type{type}
{
    Expect::NotNull(buffer);
}
