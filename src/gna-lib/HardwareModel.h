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

#include <memory>
#include <vector>

#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "HardwareRequest.h"
#include "DriverInterface.h"
#include "Memory.h"

#include "gna2-common-impl.h"

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
    static uint32_t CalculateDescriptorSize(const uint32_t layerCount,
        const uint32_t gmmLayersCount, const DeviceVersion hwId = DefaultDeviceVersion);

    HardwareModel(const std::vector<std::unique_ptr<Layer>>& layers, uint32_t gmmCount,
        const HardwareCapabilities& hwCaps);

    ~HardwareModel() = default;
    HardwareModel(const HardwareModel &) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;

    void Build(const std::vector<Memory* >& memoryObjects);

    const HardwareLayer* GetLayer(uint32_t layerIndex) const
    {
        return hardwareLayers.at(layerIndex).get();
    }

    uint64_t GetMemoryId(const BaseAddress& address) const;

    /* Calculates offset proper for GNA hardware
     * Few assumptions here:
     * a) MMU is enabled
     * b) layer descriptor memory is added first to MMU
     * c) other memory buffers are added to MMU in order they are provided
     */
    virtual uint32_t GetBufferOffset(const BaseAddress& address) const;

protected:
    static uint32_t getLayerDescriptorsSize(const uint32_t layerCount,
        DeviceVersion hwId = DefaultDeviceVersion);
    static uint32_t getGmmDescriptorsSize(const uint32_t gmmLayersCount,
        DeviceVersion hwId = DefaultDeviceVersion);

    virtual void allocateLayerDescriptors();

    std::unique_ptr<LayerDescriptor> baseDescriptor;

    const std::vector<std::unique_ptr<Layer>>& softwareLayers;
    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

    const HardwareCapabilities& hwCapabilities;

    const uint32_t gmmDescriptorsSize;
    const uint32_t xnnDescriptorsSize;

    std::unique_ptr<Memory> ldMemory;
    std::vector<Memory *> modelMemoryObjects;

    uint32_t modelSize = 0;
};

}
