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

#include "ParameterLimits.h"

#include <map>

namespace GNA
{

struct DataMode
{
    template<typename T> static T ToSize(const gna_data_mode);

    DataMode() = delete;
    DataMode(const DataMode&) = default;
    DataMode(const gna_data_mode dataMode);
    DataMode(const uint32_t dataMode);
    ~DataMode() = default;

    operator gna_data_mode() const
    {
        return Value;
    }

    // TODO:3: New API 2.1
    // const gna_data_type Type; // (e.g. uint32_t)
    // const gna_data_mode Mode; // (e.g. Disabled, const,...)

    const gna_data_mode Value;

    // Size on data element in bytes
    const uint32_t Size;

protected:
    static const std::map<const gna_data_mode, const uint32_t>& GetSizes();
};

bool operator ==(const gna_data_mode &left, const DataMode &right);

bool operator !=(const gna_data_mode &left, const DataMode &right);

using DataModeLimits = SetLimits<DataMode>;

}
