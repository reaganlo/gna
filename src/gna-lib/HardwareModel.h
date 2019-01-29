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

#include <vector>

#include "HardwareLayer.h"
#include "HardwareRequest.h"
#include "IoctlSender.h"
#include "Memory.h"

namespace GNA
{

class SoftwareModel;
class Memory;
class AccelerationDetector;
class Layer;
struct LayerConfiguration;
class RequestConfiguration;
struct RequestProfiler;

class HardwareModel
{
public:
    static size_t CalculateDescriptorSize(const uint32_t layerCount, const uint16_t gmmLayersCount);

    HardwareModel(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers, uint32_t gmmCount,
        const uint64_t memoryId, const BaseAddress memoryBase, const BaseAddress baseDescriptorAddress,  IoctlSender &sender, const AccelerationDetector& detector);
    ~HardwareModel() = default;
    HardwareModel(const HardwareModel &) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;

    const HardwareLayer* GetLayer(uint32_t layerIndex) const
    {
        return hardwareLayers.at(layerIndex).get();
    }

    inline uint32_t GetOffset(const BaseAddress& address) const
    {
        return address.GetOffset(memoryBase);
    }

    void Build();
    void InvalidateConfig(gna_request_cfg_id configId);

    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers,
        const GnaOperationMode operationMode);

protected:
    static uint32_t getLayerDescriptorsSize(const uint32_t layerCount);

    // needed for driver communication
    const uint64_t memoryId;
    const BaseAddress memoryBase;
    const gna_model_id modelId;
    LayerDescriptor baseDescriptor;
    IoctlSender &ioctlSender;
    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

private:
    static uint32_t getGmmDescriptorsSize(const uint32_t gmmLayersCount);

    std::map<gna_request_cfg_id, std::unique_ptr<HardwareRequest>> hardwareRequests;

    const std::vector<std::unique_ptr<Layer>>& softwareLayers;
    const uint32_t gmmDescriptorsSize;
};

}
