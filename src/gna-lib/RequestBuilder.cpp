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

#include "RequestBuilder.h"

#include "RequestConfiguration.h"
#include "LayerConfiguration.h"

using std::make_unique;

using namespace GNA;

gna_request_cfg_id RequestBuilder::assignConfigId()
{
    return configIdSequence++; // TODO:3: add unique id
}

void RequestBuilder::CreateConfiguration(CompiledModel& model, gna_request_cfg_id *configId, DeviceVersion consistentDevice)
{
    *configId = assignConfigId();
    configurations.emplace(*configId, make_unique<RequestConfiguration>(model, *configId, consistentDevice));
}

void RequestBuilder::ReleaseConfiguration(gna_request_cfg_id configId)
{
    //TODO:3: consider adding thread safty mechanism
    configurations.erase(configId);
}

void RequestBuilder::AttachBuffer(gna_request_cfg_id configId, GnaComponentType type, uint32_t layerIndex,
    void * address) const
{
    auto& configuration = GetConfiguration(configId);
    configuration.AddBuffer(type, layerIndex, address);
}

void RequestBuilder::AttachActiveList(gna_request_cfg_id configId, uint32_t layerIndex,
    const ActiveList& activeList) const
{
    auto& configuration = GetConfiguration(configId);
    configuration.AddActiveList(layerIndex, activeList);
}

// TODO:3: RequestBuilder inconsistent usage, some request methods are called directly, some via RequestBuilder

RequestConfiguration& RequestBuilder::GetConfiguration(gna_request_cfg_id configId) const
{
    try
    {
        auto& config = configurations.at(configId);
        return *config.get();
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

std::unique_ptr<Request> RequestBuilder::CreateRequest(gna_request_cfg_id configId)
{
    auto profiler = std::make_unique<RequestProfiler>();
    profilerTscStart(&profiler->preprocess);
    auto& configuration = GetConfiguration(configId);
    return std::make_unique<Request>(configuration, move(profiler));
}
