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

#include <memory>
#include <vector>

#include "common.h"
#include "RequestConfiguration.h"
#include "CompiledModel.h"

namespace GNA
{
class RequestBuilder
{
public:
    RequestBuilder() = default;
    RequestBuilder(const RequestBuilder &) = delete;
    RequestBuilder& operator=(const RequestBuilder&) = delete;

    void CreateConfiguration(const CompiledModel& model, gna_request_cfg_id *configId);
    void AttachBuffer(gna_request_cfg_id configId, gna_buffer_type type, uint16_t layerIndex, void * address) const;
    void AttachActiveList(gna_request_cfg_id configId, uint16_t layerIndex, const ActiveList& activeList) const;
    RequestConfiguration& GetConfiguration(gna_request_cfg_id configId) const;

private:
    std::vector<std::unique_ptr<RequestConfiguration>> configurationVector;
    gna_request_cfg_id assignConfigId();
 
    gna_request_cfg_id configIdSequence = 0;
};

}
