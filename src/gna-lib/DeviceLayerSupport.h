/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include <map>
#include <memory>

#include "common.h"
#include "Expect.h"

namespace GNA
{

typedef std::map<gna_api_version, bool> ApiSupport;
typedef std::map<gna_device_generation, bool> HwSupport;

class Support
{
public:
    Support() :
        Api{},
        Hw{}
    {
    }

    Support(ApiSupport api, HwSupport hw) :
        Api{api},
        Hw{hw}
    {
        // FIXME: change to InSet
        //Expect::InRange<size_t>(api.size(), GNA_API_NOT_SUPPORTED, GNA_API_VERSION_COUNT, Gna2StatusXnnErrorLyrCfg);
        //Expect::InRange<size_t>(hw.size(), GNA_DEVICE_NOT_SUPPORTED, GNA_DEVICE_COUNT, Gna2StatusXnnErrorLyrCfg);
        for (auto apiSupport : Api)
        {
            Expect::True(apiSupport.second, Gna2StatusNullArgumentRequired);
        }
        for (auto hwSupport : Hw)
        {
            Expect::True(hwSupport.second, Gna2StatusNullArgumentRequired);
        }
    }

    ~Support() = default;

    bool IsSupported() const;

    const ApiSupport Api;
    const HwSupport Hw;
};

struct DataConfig
{
    DataConfig(gna_data_mode input, gna_data_mode weight, gna_data_mode bias, gna_data_mode output) :
        Input{input},
        Weight{weight},
        Bias{bias},
        Output{output}
    {
    }
    ~DataConfig() = default;

    bool operator<(const DataConfig &mode) const
    {
        if (mode.Input != Input)
        {
            return mode.Input < Input;
        }

        if (mode.Weight != Weight)
        {
            return mode.Weight < Weight;
        }

        if (mode.Bias != Bias)
        {
            return mode.Bias < Bias;
        }

        if (mode.Output != Output)
        {
            return mode.Output < Output;
        }

        return false;
    }

    const gna_data_mode Input;

    union
    {
        const gna_data_mode Covariance;
        const gna_data_mode Weight;
    };

    union
    {
        const gna_data_mode Const;
        const gna_data_mode Bias;
    };

    const gna_data_mode Output;

    static const std::map<const DataConfig, std::map<const gna_layer_operation, const Support>> Capabilities;
};

}
