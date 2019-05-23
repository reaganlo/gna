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

#include "HardwareModel.h"

#include "Address.h"
#include "HardwareCapabilities.h"

#include "KernelArguments.h"

#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{
class LayerDescriptor;

class HardwareModelSue1 : public HardwareModel
{
public:
    HardwareModelSue1(
        const std::vector<std::unique_ptr<Layer>>& layers,
        uint32_t gmmCount, std::unique_ptr<Memory> dumpMemory);

    virtual ~HardwareModelSue1() = default;

    const LayerDescriptor& GetDescriptor(uint32_t layerIndex) const;

    uint32_t GetOutputOffset(uint32_t layerIndex) const;

    uint32_t GetInputOffset(uint32_t layerIndex) const;

    // this override does not add PAGE_SIZE alignment to calculations
    // since memory buffers are copied to one allocated memory buffer
    virtual uint32_t GetBufferOffset(const BaseAddress& address) const override;

protected:
    virtual void allocateLayerDescriptors() override;

private:
    static HardwareCapabilities sueCapabilities;
};

}
