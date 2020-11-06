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

static const RangeLimits<> & limitsForInputEqual1()
{
    static const RangeLimits<> _limitsForInputEqual1 =
    {
        1,
        1,
        1,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputEqual1;
}

static const RangeLimits<> & limitsForOutputEqual1()
{
    static const RangeLimits<> _limitsForOutputEqual1 =
    {
        limitsForInputEqual1(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutputEqual1;
}

static const RangeLimits<>& limitsForBiasEqual1()
{
    static const RangeLimits<> _limitsForOutputEqual1 =
    {
        limitsForInputEqual1(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForOutputEqual1;
}

static const RangeLimits<> & limitsForInputUInt16Max1D()
{
    static const RangeLimits<> _limitsForInputUInt16Max =
    {
        1,
        LayerCapabilities::InputElementCountMax,
        8,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputUInt16Max;
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
        { GNA_INT32 },
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesGen0_9;
}

static const DataModeLimits & _ModesGen3Cnn2D()
{
    static const DataModeLimits __ModesGen3Cnn2D =
    {
        { GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_DISABLED },
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesGen3Cnn2D;
}

static const RangeLimits<> & limitsForFilterNumber()
{
    static const RangeLimits<> _limits =
    {
        ConvolutionalLayer2DCapabilities::Filter1DElementsMultiplier,
        ConvolutionalLayer2DCapabilities::Filter1DCountMax,
        ConvolutionalLayer2DCapabilities::Filter1DElementsMultiplier,
        Gna2StatusCnnErrorConvFltCount
    };
    return _limits;
}

static const RangeLimits<> & limitsForOutputDepth()
{
    static const RangeLimits<> _limits =
    {
        limitsForFilterNumber(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limits;
}

const FullCapabilitiesMap & ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                {Gna2DeviceGeneration1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},    // N = 1
                    {{GNA_DIM_H, limitsForInputEqual1()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, Gna2DeviceGeneration0_9)})},
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},    // N = 1
                    {{GNA_DIM_N, limitsForInputEqual1()},
                    {GNA_DIM_H, limitsForInput()},
                    {GNA_DIM_W, limitsForInput()},
                    {GNA_DIM_D, limitsForInput()}},
                    GetModes(InputOperandIndex, Gna2DeviceGeneration3_0)})}
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},    // N = 1
                    {{GNA_DIM_N, limitsForInputEqual1()},
                    {GNA_DIM_H, limitsForInputEqual1()},
                    {GNA_DIM_W, limitsForInputUInt16Max1D()},
                    {GNA_DIM_D, limitsForInputEqual1()}},
                    {{GNA_INT16}, Gna2StatusXnnErrorInputBytes}})}
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                {Gna2DeviceGeneration1_0, std::make_shared<TensorLimits>(TensorLimits{
                { GNA_TENSOR_NWD },
                {{GNA_DIM_N, limitsForOutputEqual1()},
                {GNA_DIM_W, limitsForOutput()},
                {GNA_DIM_D, limitsForOutputDepth()}},
                GetModes(OutputOperandIndex, Gna2DeviceGeneration0_9)})},
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},
                    {{GNA_DIM_N, limitsForOutput()},
                    {GNA_DIM_H, limitsForOutput()},
                    {GNA_DIM_W, limitsForOutput()},
                    {GNA_DIM_D, {1,
                        Filter2DCountMax /* bigger limit to workaround lack of 1D/2D differentiation */,
                        1, Gna2StatusXnnErrorOutputVolume}}},
                    GetModes(OutputOperandIndex, Gna2DeviceGeneration3_0)})}
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},
                    {{GNA_DIM_N, limitsForOutputEqual1()},
                    {GNA_DIM_H, limitsForOutputEqual1()},
                    {GNA_DIM_W, limitsForOutput()},
                    {GNA_DIM_D, {1, Filter2DCountMax, 1, Gna2StatusXnnErrorOutputVolume}}},
                    {{GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes}})}
            }},
        }},
        {FilterOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                {Gna2DeviceGeneration1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NW},    // N - # filters, W - # filter coefficients
                    {{GNA_DIM_N, limitsForFilterNumber()},
                    {GNA_DIM_W, {Filter1DElementsMin, Filter1DElementsMax, shapeLimitMultipliersForCnnLegacy(), Gna2StatusCnnErrorConvFltVolume}}},
                    {{ GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})}
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {1, Filter2DCountMax, 1, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {Filter2DElementsMin, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {Filter2DElementsMin, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {Filter2DElementsMin, 2048, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ GNA_INT8, GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})},
                {Gna2DeviceGeneration3_5, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {1, Filter2DCountMax, 1, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {Filter2DElementsMin, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {Filter2DElementsMin, 4096, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {Filter2DElementsMin, 2048, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ GNA_INT8, GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {Filter1DElementsMultiplier, Filter2DCountMax, Filter1DElementsMultiplier, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {Filter2DElementsMin, Filter2DElementsMin, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {Filter1DElementsMin, Filter1DElementsMax, Filter1DElementsMin, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {Filter2DElementsMin, Filter2DElementsMin, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_CONVOLUTIONAL, {
                {Gna2DeviceGeneration1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_N},          // H - #kernel (GNA_BIAS_PER_KERNEL)
                    {{GNA_DIM_N, {Filter1DElementsMultiplier, Filter1DCountMax, Filter1DElementsMultiplier, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen0_9()})},
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                        {{GNA_DIM_N, {1, Filter1DCountMax, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, {1, LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_W, {1, LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen3Cnn2D()})}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                {Gna2DeviceGeneration3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                        {{GNA_DIM_N, {1, Filter1DCountMax, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, limitsForBiasEqual1()},
                        {GNA_DIM_W, limitsForBiasEqual1()}},
                    _ModesGen0_9()})}
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
