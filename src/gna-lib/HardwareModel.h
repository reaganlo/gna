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
#include "IoctlSender.h"
#include "Memory.h"

namespace GNA
{

class SoftwareModel;
class Memory;
class AccelerationDetector;
class Layer;
class LayerConfiguration;
class RequestConfiguration;
struct RequestProfiler;

enum GnaOperationMode : uint8_t
{
    GMM = 0,
    xNN = 1
};

class HardwareModel
{
public:
    static const size_t CalculateDescriptorSize(const uint16_t layerCount, const uint16_t gmmLayersCount);

    HardwareModel(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers, uint16_t gmmCount,
        const uint64_t memoryId, const BaseAddressC memoryBase, const BaseAddressC descriptorBase,  IoctlSender &sender, const AccelerationDetector& detector);
    ~HardwareModel() = default;
    HardwareModel(const HardwareModel &) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;

    inline uint32_t GetOffset(const BaseAddressC& address) const
    {
        return address.GetOffset(memoryBase);
    }

    void Build();

    void InvalidateConfigCache(gna_request_cfg_id configId);

    virtual status_t HardwareModel::Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers,
        const GnaOperationMode operationMode);

protected:
    static uint32_t getLayerDescriptorsSize(const uint16_t layerCount);

    // needed for driver communication
    const uint64_t memoryId;
    const BaseAddressC memoryBase;
    const gna_model_id modelId;
    BaseAddressC descriptorsAddress;
    uint32_t layerDescriptorsSize;
    const uint32_t hardwareBufferSize;
    IoctlSender &ioctlSender;
    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

private:
    static uint32_t getGmmDescriptorsSize(const uint16_t gmmLayersCount);

    size_t calculateCacheSize(uint32_t buffersCount, uint32_t nnopLayersCount, uint32_t activeListCount) const;

    void getHwConfigData(void* &buffer, size_t &size, uint16_t layerIndex, uint16_t layerCount,
        const RequestConfiguration& requestConfiguration, const GnaOperationMode operationMode) const;

    void writeBuffersIntoCache(void* &lyrsCfg, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const;
    void writeNnopTypesIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const;
    void writeXnnActiveListsIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const;
    void writeGmmActiveListsIntoCache(void* &buffer, const std::map<uint32_t, std::unique_ptr<LayerConfiguration>>& layerConfigurations) const;

    mutable std::map<gna_request_cfg_id, std::map<uint16_t, bool>> activeLists;
    mutable std::map<gna_request_cfg_id, std::unique_ptr<uint8_t[]>> requestHwCaches;
    mutable std::map<gna_request_cfg_id, size_t> requestCacheSizes;

    const std::vector<std::unique_ptr<Layer>>& softwareLayers;
    const uint32_t gmmDescriptorsSize;
};

}
