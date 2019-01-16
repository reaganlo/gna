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

#include "common.h"

#include "GnaException.h"

#include <map>
#include <vector>

namespace GNA
{

template<typename T>
struct ValueLimits
{
    T Value;
    status_t Error;
};

using AlignLimits = ValueLimits<uint32_t>;

struct BufferLimits : public AlignLimits
{
    BufferLimits(const uint32_t alignment, const status_t error) :
        ValueLimits{alignment, error}
    {}
};

template<typename T>
struct SetLimits : public std::vector<T>
{
    SetLimits(const std::vector<T>& validValuesSet, status_t error) :
        std::vector<T>{validValuesSet},
        Error{error}
    {}

    status_t Error;
};

template<typename T>
struct RangeLimits
{
    RangeLimits(T min, status_t minError, T max, status_t maxError, const SetLimits<T>& multipliers) :
        Min{min, minError},
        Max{max, maxError},
        Multipliers{multipliers}
    {
        if (Multipliers.size() <= 0)
            throw GnaException(XNN_ERR_INPUT_VOLUME);
    }

    RangeLimits(T min, status_t minError, T max, status_t maxError, T multiplier, status_t multiplierError) :
        RangeLimits{min, minError, max, maxError,
            SetLimits<T>{std::vector<T>{multiplier}, multiplierError}}
    {}

    RangeLimits(T min, T max, T multiplier, status_t error) :
        RangeLimits{min, error, max, error, multiplier, error}
    {}

    RangeLimits(T min, T max, status_t rangeError, T multiplier, status_t multiplierError) :
        RangeLimits{min, rangeError, max, rangeError, multiplier, multiplierError}
    {}

    RangeLimits(T min, T max, const std::vector<T>& multipliers, status_t error) :
        RangeLimits{min, error, max, error, SetLimits<T>{multipliers, error}}
    {}

    ValueLimits<T> Min;
    ValueLimits<T> Max;
    // multipliers for different data sizes
    // first (index 0) is effective multiplier (either set by component based on mode or the only one)
    SetLimits<T> Multipliers;

};

using ShapeLimits = std::map<const gna_tensor_dim, RangeLimits<uint32_t>>;

}
