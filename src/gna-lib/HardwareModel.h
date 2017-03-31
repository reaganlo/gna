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

#include "AccelerationDetector.h"
#include "common.h"
#include "GnaException.h"
#include "IoctlSender.h"
#include "Memory.h"
#include "SoftwareModel.h"

namespace GNA
{


class HardwareModel : protected IoctlSender
{
public:
    static const size_t CalculateDescriptorSize(const uint16_t layerCount, const uint16_t gmmLayersCount);

    HardwareModel(const gna_model_id modId, const SoftwareModel& model, const Memory& wholeMemory,
        const AccelerationDetector& detector);

    ~HardwareModel();

    uint32_t GetOffsetToBase(void* address);

private:
    void build(const std::vector<std::unique_ptr<Layer>>& layers, const uint32_t hardwareInternalBufferSize);

    static uint32_t getLayerDescriptorsSize(const uint16_t layerCount)
    {
        Expect::InRange(layerCount, 1, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
        auto layerDescriptorsSize = size_t{ layerCount * sizeof(XNN_LYR) };
        return layerDescriptorsSize;
    }
    
    static uint32_t getGmmDescriptorsSize(const uint16_t gmmLayersCount)
    {
        Expect::InRange(gmmLayersCount, 0, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
        auto gmmDescriptorsSize = size_t{ gmmLayersCount * sizeof(GMM_CONFIG) };
        return gmmDescriptorsSize;
    }

    void mapMemory(const Memory& memory);
    void unmapMemory();

    // needed for driver communication
    gna_model_id modelId;

    const BaseAddressC memoryBaseAddress;
    const uint32_t layerDescriptorsSize;
    const uint32_t gmmDescriptorsSize;

    bool memoryMapped = false;
};

}
