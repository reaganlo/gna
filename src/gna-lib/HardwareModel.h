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

#include <vector>

#include "IoctlSender.h"
#include "HardwareLayer.h"

namespace GNA
{

class SoftwareModel;
class Memory;
class AccelerationDetector;
class Layer;
class RequestConfiguration;
struct RequestProfiler;

class HardwareModel
{
public:
    static const size_t CalculateDescriptorSize(const uint16_t layerCount, const uint16_t gmmLayersCount);

    HardwareModel(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers, const Memory& wholeMemory,
        const AccelerationDetector& detector);
    ~HardwareModel() = default;
    HardwareModel(const HardwareModel &) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;

    virtual status_t HardwareModel::Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers);

    inline uint32_t GetOffset(const BaseAddressC& address) const
    {
        return address.GetOffset(memoryBaseAddress);
    }

    void WriteLayerInputBuffer(const uint32_t layerIndex, PGNA_BUFFER_DESCR &lyrsCfg,
        const ConfigurationBuffer * const buffer) const
    {
        hardwareLayers.at(layerIndex)->WriteInputBuffer(lyrsCfg, buffer);
    }
    void WriteLayerOutputBuffer(const uint32_t layerIndex, PGNA_BUFFER_DESCR &lyrsCfg,
        const ConfigurationBuffer * const buffer) const
    {
        hardwareLayers.at(layerIndex)->WriteOutputBuffer(lyrsCfg, buffer);
    }
    void WriteLayerNnopType(const uint32_t layerIndex, PNNOP_TYPE_DESCR nnopCfg, bool actListEnabled) const
    {
        hardwareLayers.at(layerIndex)->WriteNnopType(nnopCfg, actListEnabled);
    }
    void WriteLayerActiveList(const uint32_t layerIndex, HardwareActiveListDescriptor & descriptor) const
    {
        hardwareLayers.at(layerIndex)->WriteActiveList(descriptor);
    }

protected:
    void build(const std::vector<std::unique_ptr<Layer>>& layers, const uint32_t hardwareInternalBufferSize);

    static uint32_t getLayerDescriptorsSize(const uint16_t layerCount);

    static uint32_t getGmmDescriptorsSize(const uint16_t gmmLayersCount);

    // needed for driver communication
    gna_model_id modelId;

    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

    const BaseAddressC memoryBaseAddress;
    const uint32_t layerDescriptorsSize;
    const uint32_t gmmDescriptorsSize;

    bool memoryMapped = false;

    IoctlSender sender;
};

}
