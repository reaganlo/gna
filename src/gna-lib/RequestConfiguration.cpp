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
#include "GnaConfig.h"
#include "CompiledModel.h"

using namespace GNA;

RequestConfiguration::RequestConfiguration(CompiledModel& model, gna_request_cfg_id configId) :
    Model{model},
    ConfigId{configId}
{ }

void RequestConfiguration::AddBuffer(gna_buffer_type type, uint32_t layerIndex, void *address)
{
    const auto& layer = *Model.GetLayer(layerIndex);
    auto layerType = layer.Config.Type;
    if (INTEL_HIDDEN == layerType 
        || (GNA_IN == type && INTEL_OUTPUT == layerType) 
        || (GNA_OUT == type && INTEL_INPUT == layerType))
    {
        throw GnaException{ XNN_ERR_LYR_TYPE };
    }

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
        throw GnaException(XNN_ERR_LYR_CFG);
    }

    Model.InvalidateConfig(ConfigId, layerConfiguration, layerIndex);
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, const ActiveList& activeList)
{
    const auto& layer = *Model.GetLayer(layerIndex);
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

    auto found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto layerConfiguration = found.first->second.get();
    Expect::Null(layerConfiguration->ActList.get());

    auto activeListPtr = ActiveList::Create(activeList);
    layerConfiguration->ActList.swap(activeListPtr);
    ++ActiveListCount;

    Model.InvalidateConfig(ConfigId, layerConfiguration, layerIndex);
}

