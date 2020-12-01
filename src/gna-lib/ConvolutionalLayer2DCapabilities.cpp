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

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>

using namespace GNA;

template<Gna2Status status = Gna2StatusXnnErrorInputVolume>
static const RangeLimits<> & limitsForInputEqual1()
{
    return LayerCapabilities::MakeLimits<1u, 1u, 1, status>();
}

static const RangeLimits<> & limitsForOutputEqual1()
{
    return limitsForInputEqual1<Gna2StatusXnnErrorOutputVolume>();
}

static const RangeLimits<>& limitsForBiasEqual1()
{
    return limitsForInputEqual1<Gna2StatusXnnErrorBiasVolume>();
}

static const RangeLimits<> & limitsForInputUInt16Max1D()
{
    return LayerCapabilities::MakeLimits<1u, LayerCapabilities::InputElementCountMax,
        8, Gna2StatusXnnErrorBiasVolume>();
}

static const MultiplierLimits & shapeLimitMultipliersForCnnLegacy()
{
    static const MultiplierLimits _shapeLimitMultipliersForCnnLegacy =
    {
        {{Gna2DataTypeInt16, LayerCapabilities::InputElementCountMultiplier }},
            Gna2StatusCnnErrorConvFltVolume
    };
    return _shapeLimitMultipliersForCnnLegacy;
}

static const DataModeLimits & _ModesGen0_9()
{
    static const DataModeLimits __ModesGen0_9 =
    {
        { Gna2DataTypeInt32 },
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesGen0_9;
}

static const DataModeLimits & _ModesGen3Cnn2D()
{
    static const DataModeLimits __ModesGen3Cnn2D =
    {
        { Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, DataMode{} },
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesGen3Cnn2D;
}

template<Gna2Status status = Gna2StatusCnnErrorConvFltCount>
static const RangeLimits<> & limitsForFilterNumber()
{
    return LayerCapabilities::MakeLimits<
        ConvolutionalLayer2DCapabilities::Filter1DElementsMultiplier,
        ConvolutionalLayer2DCapabilities::Filter1DCountMax,
         ConvolutionalLayer2DCapabilities::Filter1DElementsMultiplier,
        status>();
}

static const RangeLimits<> & limitsForOutputDepth()
{
    return limitsForFilterNumber<Gna2StatusXnnErrorOutputVolume>();
}

const FullCapabilitiesMap & ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                LayerCaps::Make<Gna2DeviceGeneration1_0, InputOperandIndex, Gna2DeviceGeneration0_9>(
                    {GNA_TENSOR_HW},    // N = 1
                    {{GNA_DIM_H, limitsForInputEqual1()},
                    {GNA_DIM_W, GetLimitsBasedOnInputLegacy<Gna2StatusXnnErrorInputVolume>()}}),
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                LayerCaps::Make<Gna2DeviceGeneration3_0, InputOperandIndex>(
                    {GNA_TENSOR_NHWD},    // N = 1
                    {{GNA_DIM_N, limitsForInputEqual1()},
                    {GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorInputVolume>()},
                    {GNA_DIM_W, GetLimitsBasedOnInput<Gna2StatusXnnErrorInputVolume>()},
                    {GNA_DIM_D, GetLimitsBasedOnInput<Gna2StatusXnnErrorInputVolume>()}})
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                LayerCaps::Make<Gna2DeviceGeneration3_0, InputOperandIndex>(
                    {GNA_TENSOR_NHWD},    // N = 1
                    {{GNA_DIM_N, limitsForInputEqual1()},
                    {GNA_DIM_H, limitsForInputEqual1()},
                    {GNA_DIM_W, limitsForInputUInt16Max1D()},
                    {GNA_DIM_D, limitsForInputEqual1()}})
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                LayerCaps::Make<Gna2DeviceGeneration1_0, OutputOperandIndex, Gna2DeviceGeneration0_9>(
                { GNA_TENSOR_NWD },
                {{GNA_DIM_N, limitsForOutputEqual1()},
                {GNA_DIM_W, GetLimitsBasedOnInput<Gna2StatusXnnErrorOutputVolume>()},
                {GNA_DIM_D, limitsForOutputDepth()}}),
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                LayerCaps::Make<Gna2DeviceGeneration3_0, OutputOperandIndex>(
                    {GNA_TENSOR_NHWD},
                    {{GNA_DIM_N, GetLimitsBasedOnInput<Gna2StatusXnnErrorOutputVolume>()},
                    {GNA_DIM_H, GetLimitsBasedOnInput<Gna2StatusXnnErrorOutputVolume>()},
                    {GNA_DIM_W, GetLimitsBasedOnInput<Gna2StatusXnnErrorOutputVolume>()},
                    {GNA_DIM_D, {1,
                        Filter2DCountMax /* bigger limit to workaround lack of 1D/2D differentiation */,
                        1, Gna2StatusXnnErrorOutputVolume}}})
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                LayerCaps::Make<Gna2DeviceGeneration3_0, OutputOperandIndex, Gna2DeviceGeneration0_9>(
                    {GNA_TENSOR_NHWD},
                    {{GNA_DIM_N, limitsForOutputEqual1()},
                    {GNA_DIM_H, limitsForOutputEqual1()},
                    {GNA_DIM_W, GetLimitsBasedOnInput<Gna2StatusXnnErrorOutputVolume>()},
                    {GNA_DIM_D, {1, Filter2DCountMax, 1, Gna2StatusXnnErrorOutputVolume}}}),
            }},
        }},
        {FilterOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                LayerCaps::Make<Gna2DeviceGeneration1_0>(
                    {GNA_TENSOR_NW},    // N - # filters, W - # filter coefficients
                    {{GNA_DIM_N, limitsForFilterNumber()},
                    {GNA_DIM_W, {Filter1DElementsMin, Filter1DElementsMax, shapeLimitMultipliersForCnnLegacy(), Gna2StatusCnnErrorConvFltVolume}}},
                    {{ Gna2DataTypeInt16 }, Gna2StatusXnnErrorConvFltBytes })
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                LayerCaps::Make<Gna2DeviceGeneration3_0>(
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {1, Filter2DCountMax, 1, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {Filter2DElementsMin, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {Filter2DElementsMin, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {Filter2DElementsMin, 2048, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ Gna2DataTypeInt8, Gna2DataTypeInt16 }, Gna2StatusXnnErrorConvFltBytes }),
                LayerCaps::Make<Gna2DeviceGeneration3_5>(
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {1, Filter2DCountMax, 1, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {Filter2DElementsMin, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {Filter2DElementsMin, 4096, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {Filter2DElementsMin, 2048, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ Gna2DataTypeInt8, Gna2DataTypeInt16 }, Gna2StatusXnnErrorConvFltBytes })
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                LayerCaps::Make<Gna2DeviceGeneration3_0>(
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {Filter1DElementsMultiplier, Filter2DCountMax, Filter1DElementsMultiplier, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {Filter2DElementsMin, Filter2DElementsMin, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {Filter1DElementsMin, Filter1DElementsMax, Filter1DElementsMin, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {Filter2DElementsMin, Filter2DElementsMin, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ Gna2DataTypeInt16 }, Gna2StatusXnnErrorConvFltBytes })
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_CONVOLUTIONAL, {
                LayerCaps::Make<Gna2DeviceGeneration1_0>(
                    {GNA_TENSOR_N},          // H - #kernel (GNA_BIAS_PER_KERNEL)
                    {{GNA_DIM_N, {Filter1DElementsMultiplier, Filter1DCountMax, Filter1DElementsMultiplier, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen0_9()),
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                LayerCaps::Make<Gna2DeviceGeneration3_0>(
                    {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                        {{GNA_DIM_N, {1, Filter1DCountMax, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, {1, InputElementCountMax, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_W, {1, InputElementCountMax, 1, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen3Cnn2D())
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                LayerCaps::Make<Gna2DeviceGeneration3_0>(
                    {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                        {{GNA_DIM_N, {1, Filter1DCountMax, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, limitsForBiasEqual1()},
                        {GNA_DIM_W, limitsForBiasEqual1()}},
                    _ModesGen0_9())
            }},
        }},
    };
    return operands.at(operandIndex);
}

const OperationCapabilityMap& ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex,
    nn_operation operation)
{
    return GetOperands(operandIndex).at(operation);
}

const FullCapabilitiesMap & ConvolutionalLayer2DCapabilities::GetParameters(uint32_t parameterIndex)
{
    static const ComponentFullCapabilityMap parameters =
    {
        {ConvolutionStrideParamIndex,{
            {INTEL_CONVOLUTIONAL_2D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}}}))},
                { Gna2DeviceGeneration3_5, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, 4096, 1, Gna2StatusCnnErrorConvFltStride}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, Filter1DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}}}))}
            }},
        }},
        {ZeroPaddingParamIndex,{
            {INTEL_CONVOLUTIONAL_2D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                        {{GNA_DIM_H, {0, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltPadding}},
                        {GNA_DIM_W, {0, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltPadding}}}))},
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {0, 0, 1, Gna2StatusCnnErrorConvFltPadding}},
                    {GNA_DIM_W, {0, 0, 1, Gna2StatusCnnErrorConvFltPadding}}}))}
            }},
        }},
        {PoolingStrideParamIndex,{
            {INTEL_CONVOLUTIONAL_2D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolStride}},
                    {GNA_DIM_W, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolStride}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorPoolStride}},
                    {GNA_DIM_W, {1, PoolingWindowSizeMax, 1, Gna2StatusCnnErrorPoolStride}}}))}
            }},
        }},
        {PoolingWindowParamIndex,{
            {INTEL_CONVOLUTIONAL_2D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolSize}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {0, PoolingWindowSizeMax, 1, Gna2StatusCnnErrorPoolSize}}}))}
            }},
        }},
        {PoolingModeParamIndex,{
        // TODO:3: add possibility to use other limits
    }},
    };
    return parameters.at(parameterIndex);
}

const OperationCapabilityMap& ConvolutionalLayer2DCapabilities::GetParameters(uint32_t parameterIndex,
    nn_operation operation)
{
    return GetParameters(parameterIndex).at(operation);
}
