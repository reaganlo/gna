/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include <cstdint>
#include <memory>

struct GNA3_AdaptHW;
struct GNA3_LyrDesc;

namespace GNA
{
struct ConvolutionFunction2D;
struct DataMode;
class PoolingFunction2D;

struct HwUarchParams
{
    bool Valid;
    uint16_t KWG;
    uint16_t KWGIter;
    uint8_t uT;
    uint8_t KMemBase;
    uint8_t CMemBase;
    uint8_t PMemBase;

    HwUarchParams() = default;
    explicit HwUarchParams(struct GNA3_AdaptHW const& source);
};

class HwModuleInterface
{
public:
    /**
     * Create HW Module for underlying OS.
     * 
     * @param moduleName Name of library without path and extension.
     */
    static std::unique_ptr<HwModuleInterface const> Create(char const* moduleName);

    HwModuleInterface(const HwModuleInterface&) = delete;
    HwModuleInterface& operator=(const HwModuleInterface&) = delete;
    virtual ~HwModuleInterface() = default;

    HwUarchParams GetCnnParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode, bool is1D) const;

protected:
    HwModuleInterface() = default;


    HwUarchParams Get1DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode) const;
    HwUarchParams Get2DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode) const;
    static int32_t GetPoolingMode(PoolingFunction2D const* poolingIn);

    typedef struct GNA3_LyrDesc* (*CreateLDFunction)();
    typedef void(*FreeLDFunction)(struct GNA3_LyrDesc* LD);
    typedef bool(*FillLDFunction)(struct GNA3_LyrDesc* LD);

    CreateLDFunction CreateLD = nullptr;
    FreeLDFunction FreeLD = nullptr;
    FillLDFunction FillLD = nullptr;

    void Validate() const;
};
}
