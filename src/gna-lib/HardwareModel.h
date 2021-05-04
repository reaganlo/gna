/*
 INTEL CONFIDENTIAL
 Copyright 2018-2020 Intel Corporation.

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
#include "HardwareLayer.h"
#include "HwModuleInterface.hpp"
#include "KernelArguments.h"
#include "LayerDescriptor.h"
#include "MemoryContainer.h"
#include "SubModel.h"


#include <cstdint>
#include <map>
#include <memory>

#include "gna2-common-impl.h"

namespace GNA
{

class CompiledModel;

class HardwareCapabilities;

class Layer;

class HardwareModel
{
public:
    HardwareModel(CompiledModel const & softwareModel, const HardwareCapabilities& hwCaps);

    HardwareModel(const HardwareModel &) = delete;
    HardwareModel& operator=(const HardwareModel&) = delete;
    virtual ~HardwareModel() = default;

    void Build(const std::vector<std::unique_ptr<SubModel>>& submodels);

    HardwareLayer const & GetLayer(uint32_t layerIndex) const;

    HardwareLayer const * TryGetLayer(uint32_t layerIndex) const;

    /* Calculates offset proper for GNA hardware
     * Few assumptions here:
     * a) MMU is enabled
     * b) layer descriptor memory is added first to MMU
     * c) other memory buffers are added to MMU in order they are provided
     */
    virtual LdaOffset GetBufferOffset(const BaseAddress& address) const;

protected:
    uint32_t calculateDescriptorSize(bool includeGmms) const;

    static uint32_t getLayerDescriptorsSize(const uint32_t layerCount,
        DeviceVersion deviceVersion = DefaultDeviceVersion);
    static uint32_t getGmmDescriptorsSize(const uint32_t gmmLayersCount);

    virtual void prepareAllocationsAndModel();

    void prepareBaseDescriptor();

    void createScratchPadMemory(void * buffer, uint32_t size);

    bool IsSoftwareLayer(const std::vector<std::unique_ptr<SubModel>>& submodels, uint32_t layerIndex);

    std::unique_ptr<LayerDescriptor> baseDescriptor;

    CompiledModel const & model;

    const HardwareCapabilities& hwCapabilities;

    const uint32_t gmmDescriptorsSize;

    const uint32_t xnnDescriptorsSize;

    std::vector<std::unique_ptr<HardwareLayer>> hardwareLayers;

    std::unique_ptr<Memory> ldMemory;

    std::unique_ptr<Memory> scratchPadMemory;

    // hardware model (ldMemory) + software model allocations
    MemoryContainer allocations;

    std::unique_ptr<HwModuleInterface const> const HwModule;

    GetHwOffset getHwOffsetFunction;
};

}
