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

using namespace GNA;

void RequestConfiguration::AddBuffer(gna_buffer_type type, uint32_t layerIndex, void *address)
{
    auto found = LayerConfigurations.find(layerIndex);
    if(found == LayerConfigurations.end())
    {
        LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    }

    auto& layerConfiguration = LayerConfigurations.at(layerIndex);
    switch(type)
    {
    case GNA_IN:
        if (layerConfiguration->InputBuffer)
            throw GnaException(GNA_ERR_UNKNOWN);
        layerConfiguration->InputBuffer = std::make_unique<ConfigurationBuffer>(GNA_IN, address);
        ++InputBuffersCount;
        break;
    case GNA_OUT:
        if (layerConfiguration->OutputBuffer)
            throw GnaException(GNA_ERR_UNKNOWN);
        layerConfiguration->OutputBuffer = std::make_unique<ConfigurationBuffer>(GNA_OUT, address);
        ++OutputBuffersCount;
        break;
    default:
        throw GnaException(GNA_ERR_UNKNOWN);
    }
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, uint32_t indicesCount, uint32_t *indices)
{
    auto found = LayerConfigurations.find(layerIndex);
    if(found == LayerConfigurations.end())
    {
        LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    }

    auto& layerConfiguration = LayerConfigurations.at(layerIndex);
    layerConfiguration->ActiveList = std::make_unique<ActiveList>(indicesCount, indices);
}

void ConfigurationBuffer::validate() const
{
    Expect::NotNull(address);
}

ConfigurationBuffer::ConfigurationBuffer(gna_buffer_type type, void* address)
    : type(type), address(address)
{
    validate();
}
