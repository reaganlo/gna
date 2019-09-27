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
#include <cstdint>

class KernelDataMode {
    uint32_t bytesPerElement;
public:
    explicit KernelDataMode(uint32_t bytesPerElementIn):bytesPerElement{ bytesPerElementIn }
    {
    }
    operator uint32_t() const
    {
        return bytesPerElement;
    }
};
// TODO:3: consider changing to class with c-tor instead of ToKernelBiasMode
enum KernelBiasMode
{
    KernelBiasModePerFilter,
    KernelBiasModePerStride,
    KernelBiasModeDisabled
};

struct ConvolutionConfig2D
{
    ConvolutionConfig2D(const uint32_t InputWidthIn, const uint32_t InputHeightIn,
        const uint32_t InputDepthIn, const uint32_t NumberOfFiltersIn,
        const uint32_t FilterWidthIn, const uint32_t FilterHeightIn,
        const uint32_t FilterDepthIn, const KernelDataMode FilterDataModeIn,
        const void* const FilterDataIn, const uint32_t StrideWidthIn,
        const uint32_t StrideHeightIn, const uint32_t ZeroPaddingWidthIn,
        const uint32_t ZeroPaddingHeightIn, const KernelBiasMode BiasModeIn,
        const KernelDataMode BiasDataModeIn, const void* const BiasDataIn) :
    InputWidth{ InputWidthIn },
        InputHeight{ InputHeightIn },
        InputDepth{ InputDepthIn },
        NumberOfFilters{ NumberOfFiltersIn },
        FilterWidth{ FilterWidthIn },
        FilterHeight{ FilterHeightIn },
        FilterDepth{ FilterDepthIn },
        FilterDataMode{ FilterDataModeIn },
        FilterData{ FilterDataIn },
        StrideWidth{ StrideWidthIn },
        StrideHeight{ StrideHeightIn },
        ZeroPaddingWidth{ ZeroPaddingWidthIn },
        ZeroPaddingHeight{ ZeroPaddingHeightIn },
        BiasMode{ BiasModeIn },
        BiasDataMode{ BiasDataModeIn },
        BiasData{ BiasDataIn }
    {
    }
    const uint32_t InputWidth;
    const uint32_t InputHeight;
    const uint32_t InputDepth;

    const uint32_t NumberOfFilters;
    const uint32_t FilterWidth;
    const uint32_t FilterHeight;
    const uint32_t FilterDepth;

    //TODO:3:P1 Check why following field is not referenced
    const KernelDataMode FilterDataMode;
    const void* const FilterData;

    const uint32_t StrideWidth;
    const uint32_t StrideHeight;

    const uint32_t ZeroPaddingWidth;
    const uint32_t ZeroPaddingHeight;

    const KernelBiasMode BiasMode;
    const KernelDataMode BiasDataMode;
    const void* const BiasData;
};
