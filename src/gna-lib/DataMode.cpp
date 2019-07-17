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

#include "DataMode.h"

#include "gna2-model-api.h"
#include "gna2-model-impl.h"

#include <cstdint>

using namespace GNA;

namespace GNA
{

template<>
const std::map<const gna_data_mode, const uint32_t>& DataMode::GetSizes()
{
    static const std::map<const gna_data_mode, const uint32_t> sizes =
    {
        {GNA_INT8, 1},
        {GNA_INT16, 2},
        {GNA_INT32, 4},
        {GNA_INT64, 8},
        {GNA_UINT8, 1},
        {GNA_UINT16, 2},
        {GNA_UINT32, 4},
        {GNA_UINT64, 8},
        {GNA_DATA_RICH_FORMAT, 8},
        {GNA_DATA_CONSTANT_SCALAR, 4},
        {GNA_DATA_ACTIVATION_DISABLED, 4},
        {GNA_DATA_DISABLED, GNA_DATA_NOT_SUPPORTED},
    };
    return sizes;
}

template<>
const std::map<const DataType, const uint32_t>& DataMode::GetSizes()
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
    return sizes;
}

}
DataType DataMode::TypeFromDataMode(const gna_data_mode dataMode)
{
    static const std::map<const gna_data_mode, const DataType> types =
    {
        {GNA_DATA_NOT_SUPPORTED, Gna2DataTypeNone},
        {GNA_INT8, Gna2DataTypeInt8},
        {GNA_INT16, Gna2DataTypeInt16},
        {GNA_INT32, Gna2DataTypeInt32},
        {GNA_INT64, Gna2DataTypeInt64},
        {GNA_UINT8, Gna2DataTypeUint8},
        {GNA_UINT16, Gna2DataTypeUint16},
        {GNA_UINT32, Gna2DataTypeUint32},
        {GNA_UINT64, Gna2DataTypeUint64},
        {GNA_DATA_RICH_FORMAT, Gna2DataTypeCompoundBias},
        {GNA_DATA_CONSTANT_SCALAR, Gna2DataTypeInt32},
        {GNA_DATA_ACTIVATION_DISABLED, Gna2DataTypeNone},
        {GNA_DATA_DISABLED, Gna2DataTypeNone},
    };
    return types.at(dataMode);
}

TensorMode DataMode::ModeFromDataMode(const gna_data_mode dataMode)
{
    static const std::map<const gna_data_mode, const TensorMode> types =
    {
        {GNA_DATA_NOT_SUPPORTED, Gna2TensorModeDisabled},
        {GNA_INT8, Gna2TensorModeDefault},
        {GNA_INT16, Gna2TensorModeDefault},
        {GNA_INT32, Gna2TensorModeDefault},
        {GNA_INT64, Gna2TensorModeDefault},
        {GNA_UINT8, Gna2TensorModeDefault},
        {GNA_UINT16, Gna2TensorModeDefault},
        {GNA_UINT32, Gna2TensorModeDefault},
        {GNA_UINT64, Gna2TensorModeDefault},
        {GNA_DATA_RICH_FORMAT, Gna2TensorModeDefault},
        {GNA_DATA_CONSTANT_SCALAR, Gna2TensorModeConstantScalar},
        {GNA_DATA_ACTIVATION_DISABLED, Gna2TensorModeDisabled},
        {GNA_DATA_DISABLED, Gna2TensorModeDisabled},
    };
    return types.at(dataMode);
}

gna_data_mode DataMode::ModeFromDataMode(const DataType dataType)
{
    static const std::map<const DataType, const gna_data_mode> types =
    {
        {Gna2DataTypeNone, GNA_DATA_DISABLED},
        {Gna2DataTypeBoolean, GNA_DATA_NOT_SUPPORTED},
        {Gna2DataTypeInt4, GNA_DATA_NOT_SUPPORTED},
        {Gna2DataTypeInt8, GNA_INT8},
        {Gna2DataTypeInt16, GNA_INT16},
        {Gna2DataTypeInt32, GNA_INT32},
        {Gna2DataTypeInt64, GNA_INT64},
        {Gna2DataTypeUint4, GNA_DATA_NOT_SUPPORTED},
        {Gna2DataTypeUint8, GNA_UINT8},
        {Gna2DataTypeUint16, GNA_UINT16},
        {Gna2DataTypeUint32, GNA_UINT32},
        {Gna2DataTypeUint64, GNA_UINT64},
        {Gna2DataTypeCompoundBias, GNA_DATA_RICH_FORMAT},
        {Gna2DataTypePwlSegment, GNA_DATA_RICH_FORMAT},
        {Gna2DataTypeWeightScaleFactor, GNA_DATA_RICH_FORMAT},
    };
    return types.at(dataType);
}

DataMode::DataMode(const gna_data_mode dataMode) :
    Value{ dataMode },
    Type{ TypeFromDataMode(dataMode) },
    Mode{ ModeFromDataMode(dataMode) },
    Size{ ToSize<uint32_t>(Value) }
{
}

DataMode::DataMode(const uint32_t dataMode) :
    DataMode(static_cast<gna_data_mode>(dataMode))
{
}

DataMode::DataMode(const DataType dataType) :
    Value{ ModeFromDataMode(dataType) },
    Type{ dataType },
    Mode{ },
    Size{ ToSize<uint32_t>(dataType) }
{
}

DataMode::DataMode(const DataType dataType, const TensorMode tensorMode) :
    Value{ ModeFromDataMode(dataType) },
    Type{ dataType },
    Mode{ tensorMode },
    Size{ ToSize<uint32_t>(dataType) }
{
}

bool GNA::operator ==(const gna_data_mode& left, const DataMode& right)
{
    return right.Value == left;
}

bool GNA::operator !=(const gna_data_mode& left, const DataMode& right)
{
    return right.Value != left;
}

bool GNA::operator ==(Gna2DataType left, const DataMode& right)
{
    return right.Type == left;
}

bool GNA::operator !=(Gna2DataType left, const DataMode& right)
{
    return right.Type != left;
}
