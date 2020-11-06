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

#include "AffineLayerCapabilities.h"

#include "DataMode.h"
#include "Capabilities.h"
#include "Tensor.h"

#include "LayerCapabilities.h"

using namespace GNA;

static const DataModeLimits& _ModesWeightGen0_9()
{
    static const DataModeLimits __ModesWeightGen0_9 =
    {
        {GNA_INT8, GNA_INT16},
        Gna2StatusXnnErrorWeightBytes
    };
    return __ModesWeightGen0_9;
}

static const DataModeLimits& _ModesBiasGen0_9()
{
    static const DataModeLimits __ModesBiasGen0_9 =
    {
        {GNA_INT32, GNA_DATA_RICH_FORMAT},
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen0_9;
}

static const DataModeLimits& _ModesBiasGen3()
{
    static const DataModeLimits __ModesBiasGen3 =
    {
        {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_RICH_FORMAT},
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen3;
}

static const DataModeLimits& _ModesBiasGen0_9Multibias()
{
    static const DataModeLimits __ModesBiasGen0_9Multibias =
    {
        {GNA_INT32 },
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen0_9Multibias;
}

static const DataModeLimits& _ModesBiasGen3Multibias()
{
    static const DataModeLimits __ModesBiasGen3Multibias =
    {
        {GNA_INT8, GNA_INT16, GNA_INT32},
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen3Multibias;
}

const RangeLimits<>& limitsForWeightMultiplierElemMaxMultiplier()
{
    static const RangeLimits<> _limitsForWeightMultiplyElemMaxMultiply =
    {
        LayerCapabilities::InputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementCountMultiplier,
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeightMultiplyElemMaxMultiply;
}

const RangeLimits<>& limitsForWeight()
{
    static const RangeLimits<> _limitsForWeight =
    {
        LayerCapabilities::limitsForInput(),
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeight;
}

const RangeLimits<>& limitsForBias()
{
    static const RangeLimits<> _limitsForBias =
    {
        LayerCapabilities::limitsForInput(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForBias;
}

static const RangeLimits<>& limitsForBiasGroupsMax()
{
    static const RangeLimits<> _limitsForBiasGroupsMax =
    {
        LayerCapabilities::limitsForInputGroupsMax(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForBiasGroupsMax;
}

static const RangeLimits<>& limitsForOutputRnn()
{
    static const RangeLimits<> _limitsForOutputRnn =
    {
        LayerCapabilities::RecurrentOutputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::RecurrentOutputElementCountMultiplier,
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutputRnn;
}

static const RangeLimits<>& limitsForWeightRnnHeight()
{
    static const RangeLimits<> _limitsForWeightRnnHeight =
    {
        limitsForOutputRnn(),
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeightRnnHeight;
}

static const RangeLimits<>& limitsForWeightRnnWidth()
{
    static const RangeLimits<> _limitsForWeightRnnBasedOnInput =
    {
        LayerCapabilities::InputElementCountMultiplier + LayerCapabilities::RecurrentOutputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax + LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementCountMultiplier,
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeightRnnBasedOnInput;
}

static const RangeLimits<>& limitsForBiasRnn()
{
    static const RangeLimits<> _limitsForBiasRnn =
    {
        limitsForOutputRnn(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForBiasRnn;
}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetInputComponentLimits(const Gna2DeviceGeneration generation)
{
    static const OperationCapabilityMap operands =
    {
        {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForInputShapeLegacy()},
            {GNA_DIM_W, limitsForInputGroupsMax()}},
            GetModes(InputOperandIndex, Gna2DeviceGeneration0_9)})},
        {Gna2DeviceGeneration2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForInputShapeLegacy()},
            {GNA_DIM_W, limitsForInputGroupsMax()}},
            GetModes(InputOperandIndex, Gna2DeviceGeneration0_9)})},
        {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForInputShapeLegacy()},
            {GNA_DIM_W, limitsForInputGroupsMax()}},
            GetModes(InputOperandIndex, Gna2DeviceGeneration3_0)})},
    };
    return operands.at(generation);
}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetOutputComponentLimits(const Gna2DeviceGeneration generation)
{
    static const OperationCapabilityMap operands =
    {
        {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForOutput()},
            {GNA_DIM_W, limitsForOutputGroupsMax()}},
            GetModes(OutputOperandIndex, Gna2DeviceGeneration0_9)})},
        {Gna2DeviceGeneration2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForOutput()},
            {GNA_DIM_W, limitsForOutputGroupsMax()}},
            GetModes(OutputOperandIndex, Gna2DeviceGeneration0_9)})},
        {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForOutput()},
            {GNA_DIM_W, limitsForOutputGroupsMax()}},
            GetModes(OutputOperandIndex, Gna2DeviceGeneration3_0)})},
    };
    return operands.at(generation);
}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetMBOutputComponentLimits(const Gna2DeviceGeneration generation)
{
    auto multiBiasLimits = GetOutputComponentLimits(Gna2DeviceGeneration2_0);
    multiBiasLimits->Dimensions.at(GNA_DIM_H).Multipliers.at(Gna2DataTypeNone) = 8;
    static const OperationCapabilityMap operands =
    {
        {Gna2DeviceGeneration2_0, multiBiasLimits},
    };
    return operands.at(generation);
}

const FullCapabilitiesMap& AffineLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_AFFINE, {
                {Gna2DeviceGeneration0_9, GetInputComponentLimits(Gna2DeviceGeneration0_9)},
                {Gna2DeviceGeneration3_0, GetInputComponentLimits(Gna2DeviceGeneration3_0)},
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {Gna2DeviceGeneration0_9, GetInputComponentLimits(Gna2DeviceGeneration0_9)},
                {Gna2DeviceGeneration3_0, GetInputComponentLimits(Gna2DeviceGeneration3_0)},
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {Gna2DeviceGeneration2_0, GetInputComponentLimits(Gna2DeviceGeneration2_0)},
                {Gna2DeviceGeneration3_0, GetInputComponentLimits(Gna2DeviceGeneration3_0)},
            }},
            {INTEL_RECURRENT, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                     GetModes(InputOperandIndex, Gna2DeviceGeneration0_9)})},
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, Gna2DeviceGeneration3_0)})},
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_AFFINE, {
                {Gna2DeviceGeneration0_9, GetOutputComponentLimits(Gna2DeviceGeneration0_9)},
                {Gna2DeviceGeneration3_0, GetOutputComponentLimits(Gna2DeviceGeneration3_0)},
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {Gna2DeviceGeneration0_9, GetOutputComponentLimits(Gna2DeviceGeneration0_9)},
                {Gna2DeviceGeneration3_0, GetOutputComponentLimits(Gna2DeviceGeneration3_0)},
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {Gna2DeviceGeneration2_0, GetMBOutputComponentLimits(Gna2DeviceGeneration2_0)},
                {Gna2DeviceGeneration3_0, GetOutputComponentLimits(Gna2DeviceGeneration3_0)},
            }},
            {INTEL_RECURRENT, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputRnn()}}, // must be multiple 32 to keep 64B output buffer alignment
                    GetModes(OutputOperandIndex, Gna2DeviceGeneration0_9)})},
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputRnn()}}, // must be multiple 32 to keep 64B output buffer alignment
                    GetModes(OutputOperandIndex, Gna2DeviceGeneration3_0)})},
            }},
        }},
        {WeightOperandIndex,{
            {INTEL_AFFINE, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},    // W - #inputs, H - #outputs
                    {{GNA_DIM_W, limitsForWeightMultiplierElemMaxMultiplier()},
                    {GNA_DIM_H, limitsForWeight()}},
                    _ModesWeightGen0_9()})}
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},    // W=H = #outputs
                    {{GNA_DIM_H, limitsForWeightMultiplierElemMaxMultiplier()}},
                    _ModesWeightGen0_9()})}
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {Gna2DeviceGeneration2_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},   // W - #inputs, H - #outputs
                    {{GNA_DIM_W, limitsForWeightMultiplierElemMaxMultiplier()},
                    {GNA_DIM_H, limitsForWeight()}},
                    _ModesWeightGen0_9()})}
            }},
            {INTEL_RECURRENT, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_HW },
                    {{GNA_DIM_H, limitsForWeightRnnHeight()},
                    {GNA_DIM_W, limitsForWeightRnnWidth()}},
                    _ModesWeightGen0_9()})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_AFFINE, {
                {Gna2DeviceGeneration0_9,std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen0_9()})},
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen3()})},
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen0_9()})},
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen3()})},
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {Gna2DeviceGeneration2_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForBias()},
                    {GNA_DIM_W, limitsForBiasGroupsMax()}},
                    _ModesBiasGen0_9Multibias()})},
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForBias()},
                    {GNA_DIM_W, limitsForBiasGroupsMax()}},
                    _ModesBiasGen3Multibias()})}
            }},
            {INTEL_RECURRENT, {
                {Gna2DeviceGeneration0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForOutputRnn()}},
                    _ModesBiasGen0_9()})},
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBiasRnn()}},
                    _ModesBiasGen3()})},
            }},
        }},
        { WeightScaleFactorOperandIndex,{
            {INTEL_AFFINE_MULTIBIAS, {
                {Gna2DeviceGeneration2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorBiasVolume}}},
            {{ GNA_DATA_RICH_FORMAT }, Gna2StatusXnnErrorBiasBytes }})}
        }},
    }}
    };

    return operands.at(operandIndex);
}
