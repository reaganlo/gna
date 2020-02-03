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

#include "LayerCapabilities.h"

using namespace GNA;

const MultiplierMap& LayerCapabilities::InputElementCountMultipliers()
{
    static auto const multipliers = MultiplierMap{
        {Gna2DataTypeInt8, 2 * InputElementCountMultiplier},
        {Gna2DataTypeInt16, 1 * InputElementCountMultiplier},
        {Gna2DataTypeInt32, InputElementCountMultiplier / 2},
    };
    return multipliers;
}

const DataModeLimits& LayerCapabilities::GetModes(uint32_t operandIndex, gna_device_generation generation)
{
    static const std::map<uint32_t, std::map<gna_device_generation, DataModeLimits>> modes =
    {
        {InputOperandIndex,
            {{GNA_0_9, {{GNA_INT16}, Gna2StatusXnnErrorInputBytes}},
            {GNA_3_0, {{GNA_INT8, GNA_INT16}, Gna2StatusXnnErrorInputBytes}},}
        },
        {OutputOperandIndex,
            {{GNA_0_9, {{GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes}},
            {GNA_3_0, {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes}},}
        },
    };
    return modes.at(operandIndex).at(generation);
}

const RangeLimits<>& LayerCapabilities::limitsForInput()
{
    static const RangeLimits<> _limitsForInput =
    {
        1,
        InputElementCountMax,
        1,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInput;
}

const RangeLimits<>& LayerCapabilities::limitsForOutput()
{
    static const RangeLimits<> _limitsForOutput =
    {
        limitsForInput(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutput;
}

const RangeLimits<>& LayerCapabilities::limitsForInputShapeLegacy()
{
    static const RangeLimits<> _limitsForInputShapeLegacy =
    {
        InputElementCountMultiplier,
        InputElementCountMax,
        InputElementCountMultipliers(),
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputShapeLegacy;
}

const RangeLimits<>& LayerCapabilities::limitsForOutputShapeLegacy()
{
    static const RangeLimits<> _limitsForOutputShapeLegacy =
    {
        limitsForInputShapeLegacy(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutputShapeLegacy;
}

const RangeLimits<>& LayerCapabilities::limitsForInputGroupsMax()
{
    static const RangeLimits<> _limitsForInputGroupsMax =
    {
        1,
        InputGroupsCountMax,
        1,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputGroupsMax;
}

const RangeLimits<>& LayerCapabilities::limitsForOutputGroupsMax()
{
    static const RangeLimits<> _limitsForInputGroupsMax =
    {
        1,
        InputGroupsCountMax,
        1,
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForInputGroupsMax;
}
