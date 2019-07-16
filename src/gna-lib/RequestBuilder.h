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

#pragma once

#include "ProfilerConfiguration.h"
#include "RequestConfiguration.h"

#include "gna2-instrumentation-api.h"

#include <memory>
#include <cstdint>
#include <unordered_map>

namespace GNA
{
class CompiledModel;
class Request;
struct ActiveList;

class RequestBuilder
{
public:
    RequestBuilder() = default;
    RequestBuilder(const RequestBuilder &) = delete;
    RequestBuilder& operator=(const RequestBuilder&) = delete;

    void CreateConfiguration(CompiledModel& model, gna_request_cfg_id *configId, DeviceVersion consistentDevice);
    void ReleaseConfiguration(gna_request_cfg_id configId);

    void AttachBuffer(gna_request_cfg_id configId, uint32_t operandIndex, uint32_t layerIndex, void * address) const;
    void AttachActiveList(gna_request_cfg_id configId, uint32_t layerIndex, const ActiveList& activeList) const;
    RequestConfiguration& GetConfiguration(gna_request_cfg_id configId) const;
    std::unique_ptr<Request> CreateRequest(gna_request_cfg_id configId);

    void CreateProfilerConfiguration(uint32_t* configId, uint32_t numberOfInstrumentationPoints, Gna2InstrumentationPoint* selectedInstrumentationPoints, uint64_t* results);
    ProfilerConfiguration& GetProfilerConfiguration(uint32_t configId) const;
    void ReleaseProfilerConfiguration(uint32_t configId);

    bool HasConfiguration(uint32_t configId) const;

private:
    std::unordered_map<uint32_t, std::unique_ptr<RequestConfiguration>> configurations;
    static gna_request_cfg_id assignConfigId();

    std::unordered_map<uint32_t, std::unique_ptr<ProfilerConfiguration>> profilerConfigurations;
    uint32_t AssignProfilerConfigId();
    gna_request_cfg_id profilerConfigIdSequence = 0;
};

}
