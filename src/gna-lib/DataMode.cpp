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

#include "DataMode.h"

#include "GnaException.h"

using namespace GNA;

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

template<typename T> T DataMode::ToSize(const gna_data_mode mode)
{
    try
    {
        return GetSizes().at(mode);
    }
    catch (const std::exception&)
    {
        throw GnaException(GNA_ERR_INVALID_DATA_MODE);
    }
}

DataMode::DataMode(const gna_data_mode dataMode) :
    Value{ dataMode },
    Size{ ToSize<uint32_t>(Value) }
{
};

DataMode::DataMode(const uint32_t dataMode) :
    DataMode(static_cast<const gna_data_mode>(dataMode))
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

