/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "gna-api-dumper.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{
class LayerDescriptor;

class HardwareModelNoMMU : public HardwareModel
{
public:

    HardwareModelNoMMU(CompiledModel const & softwareModel, intel_gna_alloc_cb customAlloc);

    virtual ~HardwareModelNoMMU() = default;

    const LayerDescriptor& GetDescriptor(uint32_t layerIndex) const;

    uint32_t GetOutputOffset(uint32_t layerIndex) const;

    uint32_t GetInputOffset(uint32_t layerIndex) const;

    static uint32_t SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex);

    // Adds BAR index at low 2 bits
    virtual uint32_t GetBufferOffset(const BaseAddress& address) const override;

    void ExportLd(void *& exportData, uint32_t & exportDataSize);

    static constexpr uint32_t MemoryTagInput = 1;
    static constexpr uint32_t MemoryTagOutput = 2;
    static constexpr uint32_t MemoryTagReadOnly = 3;

    static constexpr uint32_t GnaDescritorSize = 32;
    static constexpr uint32_t BarIndexGnaBar = 0;
    static constexpr uint32_t BarIndexRo = 1;
    static constexpr uint32_t BarIndexInput = 2;
    static constexpr uint32_t BarIndexOutput = 3;

protected:
    virtual void allocateLayerDescriptors() override;

private:
    static HardwareCapabilities noMMUCapabilities;

    intel_gna_alloc_cb customAlloc = nullptr;

    std::unique_ptr<Memory> guessedInput;
    std::unique_ptr<Memory> guessedOutput;

    MemoryContainer ROAllocations;
    MemoryContainer InputAllocations;
    MemoryContainer OutputAllocations;
};

}
