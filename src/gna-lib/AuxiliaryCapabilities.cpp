/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "AuxiliaryCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "Tensor.h"


using namespace GNA;

static const DataModeLimits& _ModesInputDeinterleaveGen0_9()
{
    static const DataModeLimits __ModesInputDeinterleaveGen0_9 =
    {
        {GNA_INT16},
        Gna2StatusXnnErrorInputBytes
    };
    return __ModesInputDeinterleaveGen0_9;
}

static const DataModeLimits& _ModesOutputCopyGen0_9()
{
    static const DataModeLimits __ModesOutputCopyGen0_9 =
    {
        {GNA_INT16},
        Gna2StatusXnnErrorOutputBytes
    };
    return __ModesOutputCopyGen0_9;
}

static const DataModeLimits& _ModesOutputCopyGen3()
{
    static const DataModeLimits __ModesOutputCopyGen3 =
    {
        {GNA_INT8, GNA_INT16},
        Gna2StatusXnnErrorOutputBytes
    };
    return __ModesOutputCopyGen3;
}

const FullCapabilitiesMap& AuxiliaryCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_COPY, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_3_0)})},
            }},
            {INTEL_INTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_3_0)})}
            }},
            {INTEL_DEINTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputShapeLegacy()},
                    {GNA_DIM_W, limitsForInputGroupsMax()}},
                    _ModesInputDeinterleaveGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputShapeLegacy()},
                    {GNA_DIM_W, limitsForInputGroupsMax()}},
                    _ModesOutputCopyGen3()})}
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_COPY, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutput()}},
                    _ModesOutputCopyGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutput()}},
                    _ModesOutputCopyGen3()})}
            }},
            {INTEL_DEINTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutput()}},
                    GetModes(OutputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H,limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutput()}},
                    GetModes(OutputOperandIndex, GNA_3_0)})},
            }},
            {INTEL_INTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputShapeLegacy()},
                    {GNA_DIM_W, limitsForOutputGroupsMax()}},
                    GetModes(OutputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(
                    TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputShapeLegacy()},
                    {GNA_DIM_W, limitsForOutputGroupsMax()}},
                    GetModes(OutputOperandIndex, GNA_3_0)})},
            }},
        }},
    };

    return operands.at(operandIndex);
}
