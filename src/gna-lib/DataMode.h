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
#include "ParameterLimits.h"

#include "gna-api-types-xnn.h"

#include <gna2-common-api.h>
#include <gna2-model-impl.h>

#include <cstdint>
#include <map>

namespace GNA
{

struct DataMode
{
    template<typename T, typename DT>  // TODO:3:API remove
    static T ToSize(const DT mode)
    {
        try
        {
            return static_cast<T>(GetSizes<DT>().at(mode));
        }
        catch (const std::exception&)
        {
            throw GnaException(Gna2StatusDataModeInvalid);
        }
    }

    DataMode() = delete;
    DataMode(const DataMode&) = default;
    DataMode(const gna_data_mode dataMode); // TODO:3:API remove
    DataMode(const uint32_t dataMode); // TODO:3:API remove
    DataMode(const DataType dataType);
    DataMode(const DataType dataType, const TensorMode tensorMode);
    ~DataMode() = default;

    operator gna_data_mode() const // TODO:3:API remove
    {
        return Value;
    }

    const gna_data_mode Value; // TODO:3:API remove

    const DataType Type;

    const TensorMode Mode;

    // Size on data element in bytes
    const uint32_t Size;

protected:
    template<typename T> // TODO:3:API remove
    static const std::map<const T, const uint32_t>& GetSizes();

    static DataType TypeFromDataMode(const gna_data_mode dataMode); // TODO:3:API remove
    static TensorMode ModeFromDataMode(const gna_data_mode dataMode); // TODO:3:API remove
    static gna_data_mode ModeFromDataMode(const DataType dataType); // TODO:3:API remove
};

bool operator ==(const gna_data_mode &left, const DataMode &right); // TODO:3:API remove

bool operator !=(const gna_data_mode &left, const DataMode &right); // TODO:3:API remove

using DataModeLimits = SetLimits<DataMode>;

}
