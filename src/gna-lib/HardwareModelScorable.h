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

#pragma once

#include "Address.h"
#include "HardwareModel.h"
#include "HardwareRequest.h"
#include "IScorable.h"

#include "KernelArguments.h"

#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <map>
#include <vector>

namespace GNA
{

class DriverInterface;
class HardwareCapabilities;
class Layer;
class Memory;
class RequestConfiguration;
class RequestProfiler;

class HardwareModelScorable : public HardwareModel, public IScorable
{
public:

    HardwareModelScorable(
        const std::vector<std::unique_ptr<Layer>>& layers, uint32_t gmmCount,
        DriverInterface &ddi, const HardwareCapabilities& hwCapsIn);
    virtual ~HardwareModelScorable() = default;

    void InvalidateConfig(gna_request_cfg_id configId);

    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers) override;

    uint32_t GetBufferOffsetForConfiguration(
        const BaseAddress& address,
        const RequestConfiguration& requestConfiguration) const;

    void ValidateConfigBuffer(
        std::vector<Memory *> configMemoryList, Memory *bufferMemory) const;

protected:
    DriverInterface &driverInterface;

    std::map<gna_request_cfg_id, std::unique_ptr<HardwareRequest>> hardwareRequests;

    virtual void allocateLayerDescriptors() override;
};

}
