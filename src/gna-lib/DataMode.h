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

#include "ParameterLimits.h"

#include <gna2-model-impl.h>

#include <cstdint>

namespace GNA
{

struct DataMode
{
    constexpr DataMode() :
        Type{ Gna2DataTypeNone },
        Mode{ Gna2TensorModeDisabled },
        Size{ 0 }
    {}
    constexpr DataMode(const DataMode &) = default;
    constexpr DataMode(DataMode &&) = default;
    DataMode(DataType type);
    DataMode(DataType type, TensorMode tensorMode);
    ~DataMode() = default;

    constexpr bool operator<(const DataMode & mode) const
    {
        if (Type != mode.Type)
        {
            return Type < mode.Type;
        }
        return Mode <= mode.Mode;
    }

    constexpr bool operator!=(const DataMode & mode) const
    {
        return Type != mode.Type || Mode != mode.Mode;
    }

    constexpr bool operator==(const DataMode & mode) const
    {
        return !(operator!=(mode));
    }

    DataMode &operator =(const DataMode & mode) = default;

    DataType Type;
    TensorMode Mode;
    // Size on data element in bytes
    uint32_t Size;

protected:
    static uint32_t GetSize(DataType type);

    static TensorMode ModeFromType(DataType type);
    static DataType TypeFromMode(DataType type, TensorMode mode);
};

using DataModeLimits = SetLimits<DataMode>;

}
