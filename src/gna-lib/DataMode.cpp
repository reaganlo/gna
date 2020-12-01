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

#include "DataMode.h"

#include "GnaException.h"

#include "gna2-model-api.h"
#include "gna2-model-impl.h"

#include <cstdint>

using namespace GNA;

uint32_t DataMode::GetSize(DataType type)
{
    try
    {
        static const std::map<const DataType, const uint32_t> sizes =
        {
            {Gna2DataTypeNone, 0},
            {Gna2DataTypeBoolean, 1},
            {Gna2DataTypeInt4, 1},
            {Gna2DataTypeInt8, 1},
            {Gna2DataTypeInt16, 2},
            {Gna2DataTypeInt32, 4},
            {Gna2DataTypeInt64, 8},
            {Gna2DataTypeUint4, 1},
            {Gna2DataTypeUint8, 1},
            {Gna2DataTypeUint16, 2},
            {Gna2DataTypeUint32, 4},
            {Gna2DataTypeUint64, 8},
            {Gna2DataTypeCompoundBias, 8},
            {Gna2DataTypePwlSegment, 8},
            {Gna2DataTypeWeightScaleFactor, 8},
        };

        return sizes.at(type);
    }
    catch (const std::exception&)
    {
        throw GnaException(Gna2StatusDataModeInvalid);
    }
}

TensorMode DataMode::ModeFromType(DataType type)
{
    try
    {
        static const std::map<const DataType, const TensorMode> types =
        {
            {Gna2DataTypeNone, Gna2TensorModeDisabled},
            {Gna2DataTypeBoolean, Gna2TensorModeDefault},
            {Gna2DataTypeInt4,  Gna2TensorModeDefault},
            {Gna2DataTypeInt8,  Gna2TensorModeDefault},
            {Gna2DataTypeInt16, Gna2TensorModeDefault},
            {Gna2DataTypeInt32, Gna2TensorModeDefault},
            {Gna2DataTypeInt64, Gna2TensorModeDefault},
            {Gna2DataTypeUint4, Gna2TensorModeDefault},
            {Gna2DataTypeUint8, Gna2TensorModeDefault},
            {Gna2DataTypeUint16, Gna2TensorModeDefault},
            {Gna2DataTypeUint32, Gna2TensorModeDefault},
            {Gna2DataTypeUint64, Gna2TensorModeDefault},
            {Gna2DataTypeCompoundBias, Gna2TensorModeDefault},
            {Gna2DataTypePwlSegment, Gna2TensorModeDefault},
            {Gna2DataTypeWeightScaleFactor, Gna2TensorModeDefault},
        };
        return types.at(type);
    }
    catch (const std::exception&)
    {
        throw GnaException(Gna2StatusDataModeInvalid);
    }
}

DataType DataMode::TypeFromMode(DataType type, TensorMode mode)
{
    switch (mode)
    {
    case Gna2TensorModeDisabled:
        return Gna2DataTypeNone;
    case Gna2TensorModeConstantScalar:
        return Gna2DataTypeInt4; // TODO:future: enhance if other than int4 constant are supported
    default:
        return type;
    }
}

DataMode::DataMode(DataType type) :
    Type{ type },
    Mode{ ModeFromType(type) },
    Size{ GetSize(Type) }
{
}

DataMode::DataMode(DataType type, TensorMode tensorMode) :
    Type{ TypeFromMode(type, tensorMode) },
    Mode{ tensorMode },
    Size{ GetSize(Type) }
{
}
