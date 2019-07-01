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

#include "ConvolutionalLayer2DCapabilities.h"


#include "Capabilities.h"
#include "DataMode.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>
#include "Tensor.h"

using namespace GNA;

static const RangeLimits<> shapeLimits_1D_input =
{1, 1, 1, Gna2StatusXnnErrorInputVolume};

static const RangeLimits<> shapeLimits_1D_output =
{shapeLimits_1D_input, Gna2StatusXnnErrorOutputVolume};

static const RangeLimits<> shapeLimits_2D_input =
{1, LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorInputVolume};

static const RangeLimits<> shapeLimits_2D_output =
{shapeLimits_2D_input, Gna2StatusXnnErrorOutputVolume};

static const RangeLimits<> shapeLimits_1D_legacy =
{
    LayerCapabilities::InputElementCountMultiplier,
    LayerCapabilities::InputElementCountMax,
    LayerCapabilities::Multipliers(),
    Gna2StatusXnnErrorInputVolume
};

const OperationCapabilityMap & ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentCapabilityMap operands =
    {
        {InputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},    // N = 1
                {{GNA_DIM_N, shapeLimits_1D_input},
                 {GNA_DIM_H, shapeLimits_1D_legacy},
                 {GNA_DIM_W, shapeLimits_1D_input},
                 {GNA_DIM_D, shapeLimits_1D_input}},
                GetModes(InputOperandIndex, GNA_0_9)})},
            {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},    // N = 1
                {{GNA_DIM_N, shapeLimits_1D_input},
                 {GNA_DIM_H, shapeLimits_2D_input},
                 {GNA_DIM_W, shapeLimits_2D_input},
                 {GNA_DIM_D, shapeLimits_2D_input}},
                GetModes(InputOperandIndex, GNA_3_0)})}
        }},
        {OutputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                { GNA_TENSOR_NHWD },
                {{GNA_DIM_N, shapeLimits_1D_output},
                {GNA_DIM_H, {Filter1DElementsMultiplier, Filter1DElementsMax, Filter1DElementsMultiplier, Gna2StatusXnnErrorOutputVolume}},
                {GNA_DIM_W, shapeLimits_2D_output},
                {GNA_DIM_D, shapeLimits_1D_output}},
                GetModes(OutputOperandIndex, GNA_0_9)})},
            {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_NHWD},
                {{GNA_DIM_N, shapeLimits_1D_output},
                 {GNA_DIM_H, shapeLimits_2D_output},
                 {GNA_DIM_W, shapeLimits_2D_output},
                 {GNA_DIM_D, shapeLimits_2D_output}},
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
                {GNA_TENSOR_WN},    // N = 1
                {{GNA_DIM_N, shapeLimits_1D_input},
                {GNA_DIM_W, shapeLimits_1D_legacy}},
                GetModes(InputOperandIndex, GNA_0_9)})},
        }},
        {OutputOperandIndex,{
            {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                { GNA_TENSOR_HN },
                {{GNA_DIM_N, shapeLimits_1D_output},
                {GNA_DIM_H, shapeLimits_2D_output}},
                GetModes(OutputOperandIndex, GNA_0_9)})},
        }},
    };
    return operands.at(operandIndex);
}

const OperationCapabilityMap & ConvolutionalLayer2DCapabilities::GetParameters(uint32_t parameterIndex)
{
    static const ComponentCapabilityMap parameters =
    {
    };
    return parameters.at(parameterIndex);
}
