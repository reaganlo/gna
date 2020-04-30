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
#include "GnaException.h"
#include "Request.h"

#include "gna-api-status.h"
#include "profiler.h"

#include <algorithm>
#include <stdexcept>

namespace GNA
{
class CompiledModel;
struct ActiveList;
}

using namespace GNA;

uint32_t RequestBuilder::assignConfigId()
{
    static uint32_t configIdSequence = 0;
    return configIdSequence++; // TODO:3: add unique id
}

void RequestBuilder::CreateConfiguration(CompiledModel& model, uint32_t *configId, DeviceVersion consistentDevice)
{
    Expect::NotNull(configId);
    *configId = assignConfigId();
    configurations.emplace(*configId, std::make_unique<RequestConfiguration>(model, *configId, consistentDevice));
}

void RequestBuilder::ReleaseConfiguration(uint32_t configId)
{
    //TODO:3: consider adding thread safty mechanism
    configurations.erase(configId);
}

void RequestBuilder::AttachBuffer(uint32_t configId, uint32_t operandIndex, uint32_t layerIndex,
    void * address) const
{
    auto& configuration = GetConfiguration(configId);
    configuration.AddBuffer(operandIndex, layerIndex, address);
}

void RequestBuilder::AttachActiveList(uint32_t configId, uint32_t layerIndex,
    const ActiveList& activeList) const
{
    auto& configuration = GetConfiguration(configId);
    configuration.AddActiveList(layerIndex, activeList);
}

// TODO:3: RequestBuilder inconsistent usage, some request methods are called directly, some via RequestBuilder

RequestConfiguration& RequestBuilder::GetConfiguration(uint32_t configId) const
{
    try
    {
        auto& config = configurations.at(configId);
        return *config;
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

std::unique_ptr<Request> RequestBuilder::CreateRequest(uint32_t configId)
{
    auto& configuration = GetConfiguration(configId);
    auto profiler = RequestProfiler::Create(configuration.GetProfilerConfiguration());
    profiler->Measure(Gna2InstrumentationPointLibPreprocessing);

    return std::make_unique<Request>(configuration, std::move(profiler));
}

uint32_t RequestBuilder::AssignProfilerConfigId()
{
    return profilerConfigIdSequence++; // TODO:3: add unique id
}

uint32_t RequestBuilder::CreateProfilerConfiguration(
    uint32_t numberOfInstrumentationPoints,
    Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t* results)
{
    auto const profilerConfigId = AssignProfilerConfigId();
    profilerConfigurations.emplace(profilerConfigId,
        std::make_unique<ProfilerConfiguration>(profilerConfigId, numberOfInstrumentationPoints, selectedInstrumentationPoints, results));
    return profilerConfigId;
}

ProfilerConfiguration& RequestBuilder::GetProfilerConfiguration(uint32_t configId) const
{
    try
    {
        auto& config = profilerConfigurations.at(configId);
        return *config.get();
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

void RequestBuilder::ReleaseProfilerConfiguration(uint32_t configId)
{
    //TODO:3: consider adding thread safety mechanism
    profilerConfigurations.erase(configId);
}

bool RequestBuilder::HasConfiguration(uint32_t configId) const
{
    return configurations.count(configId) > 0;
}
