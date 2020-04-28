/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "CompiledModel.h"
#include "Request.h"
#include "RequestConfiguration.h"

#include <algorithm>
#include <cstring>
#include <memory>

struct KernelBuffers;

using namespace GNA;

Request::Request(RequestConfiguration& config, std::unique_ptr<RequestProfiler> profiler) :
    Configuration(config),
    Profiler{std::move(profiler)}
{
    auto callback = [&](KernelBuffers *buffers, RequestProfiler *profilerPtr)
    {
        return Configuration.Model.Score(Configuration, profilerPtr, buffers);
    };
    scoreTask = std::packaged_task<Gna2Status(KernelBuffers *buffers, RequestProfiler *profiler)>(callback);
}

std::future<Gna2Status> Request::GetFuture()
{
    return scoreTask.get_future();
}

RequestProfiler::RequestProfiler(bool initialize)
{
    if (initialize)
    {
        Points.resize(ProfilerConfiguration::GetMaxNumberOfInstrumentationPoints(), 0);
    }
}

void RequestProfiler::AddResults(Gna2InstrumentationPoint point, uint64_t result)
{
    Points.at(point) += result;
}

void MillisecondProfiler::Measure(Gna2InstrumentationPoint pointType)
{
    Points.at(pointType) = static_cast<uint64_t>(std::chrono::duration_cast<chronoMs>(chronoClock::now().time_since_epoch()).count());
}

void MicrosecondProfiler::Measure(Gna2InstrumentationPoint pointType)
{
    Points.at(pointType) = static_cast<uint64_t>(std::chrono::duration_cast<chronoUs>(chronoClock::now().time_since_epoch()).count());
}

void CycleProfiler::Measure(Gna2InstrumentationPoint pointType)
{
    getTsc(&Points.at(pointType));
}

void RequestProfiler::SaveResults(ProfilerConfiguration* config)
{
    for (auto i = 0u; i < config->NPoints; i++)
    {
        config->SetResult(i, Points.at(config->Points[i]));
    }
}

std::unique_ptr<RequestProfiler> RequestProfiler::Create(ProfilerConfiguration* config)
{
    if (nullptr == config)
    {
        return std::make_unique<DisabledProfiler>();
    }

    switch (config->GetUnit())
    {
    case Gna2InstrumentationUnitMicroseconds:
        return std::make_unique<MicrosecondProfiler>();
    case Gna2InstrumentationUnitMilliseconds:
        return std::make_unique<MillisecondProfiler>();
    case Gna2InstrumentationUnitCycles:
        return std::make_unique<CycleProfiler>();
    default:
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

uint64_t RequestProfiler::ConvertElapsedTime(uint64_t frequency, uint64_t multiplier,
    uint64_t start, uint64_t stop)
{
    auto const elapsedCycles = stop - start;
    auto const round = frequency / 2;
    auto elapsedMicroseconds = (elapsedCycles * multiplier + round) / frequency;
    return elapsedMicroseconds;
}

void DisabledProfiler::Measure(Gna2InstrumentationPoint point)
{
    UNREFERENCED_PARAMETER(point);
}
void DisabledProfiler::AddResults(Gna2InstrumentationPoint point, uint64_t result)
{
    UNREFERENCED_PARAMETER(point);
    UNREFERENCED_PARAMETER(result);
}
void DisabledProfiler::SaveResults(ProfilerConfiguration* config)
{
    UNREFERENCED_PARAMETER(config);
}
