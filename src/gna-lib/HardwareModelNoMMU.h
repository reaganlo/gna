/*
 INTEL CONFIDENTIAL
 Copyright 2019-2020 Intel Corporation.

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
#include "ModelExportConfig.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{
class LayerDescriptor;

class HardwareModelNoMMU : public HardwareModel
{
public:

    HardwareModelNoMMU(CompiledModel const & softwareModel, Gna2UserAllocator customAlloc, Gna2DeviceVersion targetDevice);

    virtual ~HardwareModelNoMMU() = default;

    const LayerDescriptor& GetDescriptor(uint32_t layerIndex) const;

    uint32_t GetOutputOffset(uint32_t layerIndex) const;

    uint32_t GetInputOffset(uint32_t layerIndex) const;

    static uint32_t SetBarIndex(uint32_t offsetFromBar, uint32_t barIndex);

    // Adds BAR index at low 2 bits
    virtual LdaOffset GetBufferOffset(const BaseAddress& address) const override;

    void ExportComponent(void *& exportData, uint32_t & exportDataSize, Gna2ModelExportComponent component);

    static constexpr uint32_t MemoryTagReadWrite = 0;
    static constexpr uint32_t MemoryTagInput = 1;
    static constexpr uint32_t MemoryTagOutput = 2;
    static constexpr uint32_t MemoryTagReadOnly = 3;
    static constexpr uint32_t MemoryTagExternalBufferInput = 4;
    static constexpr uint32_t MemoryTagExternalBufferOutput = 5;
    static constexpr uint32_t MemoryTagScratch = 6;
    static constexpr uint32_t MemoryTagState = 7;

    static constexpr uint32_t GNADSCBARIndex = 0;
    static constexpr uint32_t BAR0Index = 1;
    static constexpr uint32_t BAR1Index = 2;
    static constexpr uint32_t BAR2Index = 3;

    static constexpr uint32_t StateBarIndex = BAR2Index;
    static constexpr uint32_t ScratchBarIndex = BAR1Index;

    static constexpr uint32_t BarIndexLda = BAR0Index;
    static constexpr uint32_t BarIndexGnaDescriptor = GNADSCBARIndex;


    static constexpr uint32_t GnaDescritorSize = 32;
    static constexpr uint32_t BarIndexInput = 2;
    static constexpr uint32_t BarIndexOutput = 3;

protected:
    virtual void prepareAllocationsAndModel() override;

private:
    void ExportLd(void *& exportData, uint32_t & exportDataSize);
    static const HardwareCapabilities& GetHwCaps(Gna2DeviceVersion targetDevice);

    static HardwareCapabilities noMMUCapabilities30;
    static HardwareCapabilities noMMUCapabilities31Anna;

    Gna2UserAllocator customUserAlloc = nullptr;

    void* customAllocSafe(uint32_t size)
    {
        Expect::NotNull((void*)customUserAlloc);
        auto o = customUserAlloc(size);
        Expect::NotNull(o, Gna2StatusResourceAllocationError);
        return o;
    }

    std::unique_ptr<Memory> guessedInput;
    std::unique_ptr<Memory> guessedOutput;

    MemoryContainer FollowingLdaAllocations;
    MemoryContainer InputAllocations;
    MemoryContainer OutputAllocations;
    MemoryContainer ExternalBufferInputAllocations;
    MemoryContainer ExternalBufferOutputAllocations;
    MemoryContainer ScratchAllocations;
    MemoryContainer StateAllocations;
};

}
