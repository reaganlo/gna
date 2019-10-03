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

#include "ConvolutionalLayer2DCapabilities.h"


#include "Capabilities.h"
#include "DataMode.h"
#include "ParameterLimits.h"
#include "Tensor.h"
#include "Validator.h"

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>

using namespace GNA;

static const RangeLimits<> limitsForInputEqual1 =
{
    1,
    1,
    1,
    Gna2StatusXnnErrorInputVolume
};

static const RangeLimits<> limitsForOutputEqual1 =
{
    limitsForInputEqual1,
    Gna2StatusXnnErrorOutputVolume
};

static const RangeLimits<> limitsForInputUInt16Max =
{
    1,
    LayerCapabilities::InputElementCountMax,
    1,
    Gna2StatusXnnErrorInputVolume
};

static const RangeLimits<> limitsForOutputUInt16Max =
{
    limitsForInputUInt16Max,
    Gna2StatusXnnErrorOutputVolume
};

static const RangeLimits<> limitsForInputShapeLegacy =
{
    LayerCapabilities::InputElementCountMultiplier,
    LayerCapabilities::InputElementCountMax,
    LayerCapabilities::InputElementCountMultipliers(),
    Gna2StatusXnnErrorInputVolume
};

const OperationCapabilityMap & ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentCapabilityMap operands =
    {
        {InputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},    // N = 1
                {{GNA_DIM_N, limitsForInputEqual1},
                 {GNA_DIM_H, limitsForInputShapeLegacy},
                 {GNA_DIM_W, limitsForInputEqual1},
                 {GNA_DIM_D, limitsForInputEqual1}},
                GetModes(InputOperandIndex, GNA_0_9)})},
            {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},    // N = 1
                {{GNA_DIM_N, limitsForInputEqual1},
                 {GNA_DIM_H, limitsForInputUInt16Max},
                 {GNA_DIM_W, limitsForInputUInt16Max},
                 {GNA_DIM_D, limitsForInputUInt16Max}},
                GetModes(InputOperandIndex, GNA_3_0)})}
        }},
        {OutputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                { GNA_TENSOR_NHWD },
                {{GNA_DIM_N, limitsForOutputEqual1},
                {GNA_DIM_H, {Filter1DElementsMultiplier, Filter1DElementsMax, Filter1DElementsMultiplier, Gna2StatusXnnErrorOutputVolume}},
                {GNA_DIM_W, limitsForOutputUInt16Max},
                {GNA_DIM_D, limitsForOutputEqual1}},
                GetModes(OutputOperandIndex, GNA_0_9)})},
            {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},
                {{GNA_DIM_N, limitsForOutputEqual1},
                 {GNA_DIM_H, limitsForOutputUInt16Max},
                 {GNA_DIM_W, limitsForOutputUInt16Max},
                 {GNA_DIM_D, limitsForOutputUInt16Max}},
                GetModes(OutputOperandIndex, GNA_3_0)})}
        }},
    };
    return operands.at(operandIndex);
}

const OperationCapabilityMap & ConvolutionalLayer2DCapabilities::GetLegacyOperands(uint32_t operandIndex)
{
    static const ComponentCapabilityMap operands =
    {
        {InputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},    // N = 1
                {{GNA_DIM_H, limitsForInputEqual1},
                {GNA_DIM_W, limitsForInputShapeLegacy}},
                GetModes(InputOperandIndex, GNA_0_9)})},
        }},
        {OutputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                { GNA_TENSOR_HW },
                {{GNA_DIM_H, limitsForOutputEqual1},
                {GNA_DIM_W, limitsForOutputUInt16Max}},
                GetModes(OutputOperandIndex, GNA_0_9)})},
        }},
    };
    return operands.at(operandIndex);
}

const OperationCapabilityMap & ConvolutionalLayer2DCapabilities::GetParameters(uint32_t parameterIndex)
{
    static const ComponentCapabilityMap parameters =
    {
        // TODO:3:caps:complete GetParameters
    };
    return parameters.at(parameterIndex);
}

const OperationCapabilityMap & ConvolutionalLayer1DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentCapabilityMap operands =
    {
        {InputOperandIndex,{
            {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},    // N = 1
                {{GNA_DIM_N, limitsForInputEqual1},
                 {GNA_DIM_H, limitsForInputEqual1},
                 {GNA_DIM_W, limitsForInputUInt16Max},
                 {GNA_DIM_D, limitsForInputEqual1}},
                GetModes(InputOperandIndex, GNA_3_0)})}
        }},
        {OutputOperandIndex,{
            {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},
                {{GNA_DIM_N, limitsForOutputEqual1},
                 {GNA_DIM_H, limitsForInputEqual1},
                 {GNA_DIM_W, limitsForOutputUInt16Max},
                {GNA_DIM_D, {1, CNN_1D_N_KERNELS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
                GetModes(OutputOperandIndex, GNA_3_0)})}
        }},
    };
    return operands.at(operandIndex);
}
