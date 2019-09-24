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

#include "GnaException.h"

#include "common.h"
#include "gna2-model-api.h"

#include <map>
#include <vector>

namespace GNA
{

template<typename T>
struct ValueLimits
{
    T Value;
    Gna2Status Error;
};

using AlignLimits = ValueLimits<uint32_t>;

struct BufferLimits : public AlignLimits
{
    BufferLimits(const uint32_t alignment, const Gna2Status error) :
        ValueLimits{alignment, error}
    {}
};

template<typename T>
struct SetLimits : public std::vector<T>
{
    SetLimits(const std::vector<T>& validValuesSet, Gna2Status error) :
        std::vector<T>{validValuesSet},
        Error{error}
    {}

    Gna2Status Error;
};

using MultiplierMap = std::map<Gna2DataType, uint32_t>;

struct MultiplierLimits : protected  MultiplierMap
{
    MultiplierLimits() = delete;

    MultiplierLimits(const MultiplierMap& multipliers, Gna2Status error) :
        MultiplierMap{multipliers},
        Error{error}
    {
        if (size() <= 0)
        {
            throw GnaException(Gna2StatusNullArgumentNotAllowed);
        }
    }


    MultiplierLimits(const MultiplierLimits& multipliers, Gna2Status error) :
        MultiplierMap{multipliers},
        Error{error}
    {
    }

    MultiplierLimits(uint32_t multiplier, Gna2Status error) :
        MultiplierMap{{{Gna2DataTypeNone, multiplier}}},
        Error{error}
    {}

    MultiplierLimits(uint32_t multiplierForInt8, uint32_t multiplierForInt16,
        uint32_t multiplierForInt32, Gna2Status error) :
        MultiplierMap{{
            {Gna2DataTypeInt8, multiplierForInt8},
            {Gna2DataTypeInt16, multiplierForInt16},
            {Gna2DataTypeInt32, multiplierForInt32},
            {Gna2DataTypeUint8, multiplierForInt8},
            {Gna2DataTypeUint16, multiplierForInt16},
            {Gna2DataTypeUint32, multiplierForInt32},
        }},
        Error{error}
    {}

    void SetEffective(Gna2DataType type)
    {
        if (Gna2DataTypeNone != type &&
            end() != find(type))
        {
            (*this)[Gna2DataTypeNone] = at(type);
        }
    }

    uint32_t GetEffective() const
    {
        return at(Gna2DataTypeNone);
    }

    Gna2Status Error;
};

template<typename T = uint32_t>
struct RangeLimits
{
    RangeLimits(T min, Gna2Status minError, T max, Gna2Status maxError, const MultiplierLimits& multipliers) :
        Min{min, minError},
        Max{max, maxError},
        Multipliers{multipliers}
    {
    }

    RangeLimits(T min, Gna2Status minError, T max, Gna2Status maxError, T multiplier, Gna2Status multiplierError) :
        RangeLimits{min, minError, max, maxError,
            MultiplierLimits{multiplier, multiplierError}}
    {}

    RangeLimits(T min, T max, T multiplier, Gna2Status error) :
        RangeLimits{min, error, max, error, multiplier, error}
    {}

    RangeLimits(T min, T max, Gna2Status rangeError, T multiplier, Gna2Status multiplierError) :
        RangeLimits{min, rangeError, max, rangeError, multiplier, multiplierError}
    {}

    RangeLimits(T min, T max, const MultiplierMap& multipliers, Gna2Status error) :
        RangeLimits{min, error, max, error, MultiplierLimits{multipliers, error}}
    {}

    RangeLimits(T min, T max, const MultiplierLimits& multipliers, Gna2Status error) :
        RangeLimits{min, error, max, error, multipliers}
    {}

    RangeLimits(RangeLimits const & base, Gna2Status error) :
        RangeLimits{base.Min.Value, error, base.Max.Value, error, MultiplierLimits(base.Multipliers, error)}
    {}

    ValueLimits<T> Min;
    ValueLimits<T> Max;
    // multipliers for different data sizes
    // first (index 0) is effective multiplier (either set by component based on mode or the only one)
    MultiplierLimits Multipliers;
};

using ShapeLimits = std::map<const gna_tensor_dim, RangeLimits<uint32_t>>;

struct OrderLimits : public ValueLimits<gna_tensor_order>
{
    OrderLimits(const gna_tensor_order order, const Gna2Status error = Gna2StatusXnnErrorLyrInvalidTensorOrder) :
        ValueLimits{ order, error }
    {}
};

struct ComponentLimits
{
    ComponentLimits(const ComponentLimits&) = default;
    ComponentLimits(const OrderLimits order, const ShapeLimits& dimensions) :
        Order{ order },
        Dimensions{ dimensions }
    {}

    OrderLimits Order;
    ShapeLimits Dimensions;
};

}
