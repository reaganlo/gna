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

namespace GNA
{

const RangeLimits<>& limitsForWeightMultiplierElemMaxMultiplier()
{
    return LayerCapabilities::MakeLimits<
        LayerCapabilities::InputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementCountMultiplier,
        Gna2StatusXnnErrorWeightVolume>();
}

template<Gna2Status status>
static const RangeLimits<>& GetLimitsBasedOnOutputRnn()
{
    return LayerCapabilities::MakeLimits<
        LayerCapabilities::RecurrentOutputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::RecurrentOutputElementCountMultiplier,
        status>();
}

static const RangeLimits<>& limitsForWeightRnnWidth()
{
    return LayerCapabilities::MakeLimits<
        LayerCapabilities::InputElementCountMultiplier + LayerCapabilities::RecurrentOutputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax + LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementCountMultiplier,
        Gna2StatusXnnErrorWeightVolume>();
}

template<>
struct ComponentCaps<BiasOperandIndex> : protected LayerCapabilities
{
    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        static const std::map<Gna2DeviceGeneration, DataModeLimits> modes =
        {
                {Gna2DeviceGeneration0_9,
                    {{Gna2DataTypeInt32, Gna2DataTypeCompoundBias}, Gna2StatusXnnErrorBiasBytes}},
                {Gna2DeviceGeneration3_0,
                    {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias}, Gna2StatusXnnErrorBiasBytes}},
                {Gna2DeviceGeneration3_5,
                    {MakeDataModesCartesian(
                    {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32,Gna2DataTypeCompoundBias}),Gna2StatusXnnErrorBiasBytes}},
        };
        return modes.at(generation);
    }
};

template<>
struct ComponentCaps<WeightOperandIndex> : protected LayerCapabilities
{
    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        static const std::map<Gna2DeviceGeneration, DataModeLimits> modes =
        {
            {Gna2DeviceGeneration0_9,
                {{Gna2DataTypeInt8, Gna2DataTypeInt16}, Gna2StatusXnnErrorWeightBytes}},
            {Gna2DeviceGeneration2_0,
                {{Gna2DataTypeInt8, Gna2DataTypeInt16}, Gna2StatusXnnErrorWeightBytes}},
        };
        return modes.at(generation);
    }
};

}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetMBOutputComponentLimits(const Gna2DeviceGeneration generation)
{
    auto multiBiasLimits = *reinterpret_cast<TensorLimits*>(
        LayerCaps::GetComponentLimits<OutputOperandIndex>(Gna2DeviceGeneration2_0).get());
    multiBiasLimits.Dimensions.at(GNA_DIM_H).Multipliers.at(Gna2DataTypeNone) = 8;
    static const OperationCapabilityMap operands =
    {
        {Gna2DeviceGeneration2_0, std::make_shared<TensorLimits>(multiBiasLimits)}
    };
    return operands.at(generation);
}

struct RnnInComponentCaps : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorInputVolume>()},
                    {GNA_DIM_W, GetLimitsBasedOnInputLegacy<Gna2StatusXnnErrorInputVolume>()}},
                GetModes(OutputOperandIndex, modeGeneration)}) };
    }
};

struct RnnOutComponentCaps : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorOutputVolume>()},
                    {GNA_DIM_W, GetLimitsBasedOnOutputRnn<Gna2StatusXnnErrorOutputVolume>()}},
                // must be multiple 32 to keep 64B output buffer alignment
                GetModes(OutputOperandIndex, modeGeneration)}) };
    }
};

const FullCapabilitiesMap& AffineLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_AFFINE, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, InputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_0, InputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_5, InputOperandIndex>(),
            }},
            {INTEL_AFFINE_DIAGONAL, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, InputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_0, InputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_5, InputOperandIndex>(),
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, InputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_0, InputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_5, InputOperandIndex>(),
            }},
            {INTEL_RECURRENT, {
                RnnInComponentCaps::Make<Gna2DeviceGeneration0_9>(),
                RnnInComponentCaps::Make<Gna2DeviceGeneration3_0>(),
                RnnInComponentCaps::Make<Gna2DeviceGeneration3_5>(),
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_AFFINE, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, OutputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_0, OutputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_5, OutputOperandIndex>(),
            }},
            {INTEL_AFFINE_DIAGONAL, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, OutputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_0, OutputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_5, OutputOperandIndex>(),
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {Gna2DeviceGeneration2_0, GetMBOutputComponentLimits(Gna2DeviceGeneration2_0)},
                LayerCaps::Make<Gna2DeviceGeneration3_0, OutputOperandIndex>(),
                LayerCaps::Make<Gna2DeviceGeneration3_5, OutputOperandIndex>(),
            }},
            {INTEL_RECURRENT, {
                RnnOutComponentCaps::Make<Gna2DeviceGeneration0_9>(),
                RnnOutComponentCaps::Make<Gna2DeviceGeneration3_0>(),
                RnnOutComponentCaps::Make<Gna2DeviceGeneration3_5>(),
            }},
        }},
        {WeightOperandIndex,{
            {INTEL_AFFINE, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, WeightOperandIndex>(
                    GNA_TENSOR_HW,
                    {{GNA_DIM_W, limitsForWeightMultiplierElemMaxMultiplier()},
                    {GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorWeightVolume>()}}),
            }},
            {INTEL_AFFINE_DIAGONAL, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, WeightOperandIndex>(
                    {GNA_TENSOR_H},    // W=H = #outputs
                    {{GNA_DIM_H, limitsForWeightMultiplierElemMaxMultiplier()}})
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0, WeightOperandIndex>(
                    {GNA_TENSOR_HW},   // W - #inputs, H - #outputs
                    {{GNA_DIM_W, limitsForWeightMultiplierElemMaxMultiplier()},
                    {GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorWeightVolume>()}})
            }},
            {INTEL_RECURRENT, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, WeightOperandIndex>(
                    { GNA_TENSOR_HW },
                    {{GNA_DIM_H, GetLimitsBasedOnOutputRnn<Gna2StatusXnnErrorWeightVolume>()},
                    {GNA_DIM_W, limitsForWeightRnnWidth()}})
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_AFFINE, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()}}),
                LayerCaps::Make<Gna2DeviceGeneration3_0, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()}}),
                LayerCaps::Make<Gna2DeviceGeneration3_5, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()}}),
            }},
            {INTEL_AFFINE_DIAGONAL, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()}}),
                LayerCaps::Make<Gna2DeviceGeneration3_0, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()}}),
                LayerCaps::Make<Gna2DeviceGeneration3_5, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()}}),
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0>(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()},
                    {GNA_DIM_W, LayerCapabilities::GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorBiasVolume>()}},
                    {{Gna2DataTypeInt32 }, Gna2StatusXnnErrorBiasBytes }),
                LayerCaps::Make<Gna2DeviceGeneration3_0>(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()},
                    {GNA_DIM_W, LayerCapabilities::GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorBiasVolume>()}},
                    {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
                            Gna2StatusXnnErrorBiasBytes}),
                LayerCaps::Make<Gna2DeviceGeneration3_5>(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorBiasVolume>()},
                    {GNA_DIM_W, LayerCapabilities::GetLimitsBasedOnInputGroupsMax<Gna2StatusXnnErrorBiasVolume>()}},
                    {MakeDataModesCartesian(
                        {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}),Gna2StatusXnnErrorBiasBytes})
            }},
            {INTEL_RECURRENT, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnOutputRnn<Gna2StatusXnnErrorBiasVolume>()}}),
                LayerCaps::Make<Gna2DeviceGeneration3_0, BiasOperandIndex>(
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, GetLimitsBasedOnOutputRnn<Gna2StatusXnnErrorBiasVolume>()}}),
            }},
        }},
        { WeightScaleFactorOperandIndex,{
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0>(
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorBiasVolume}}},
            {{ Gna2DataTypeWeightScaleFactor }, Gna2StatusXnnErrorBiasBytes })
        }},
    }}
    };

    return operands.at(operandIndex);
}
