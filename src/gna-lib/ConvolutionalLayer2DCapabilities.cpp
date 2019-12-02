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

static const RangeLimits<> & limitsForInputUInt16Max()
{
    static const RangeLimits<> _limitsForInputUInt16Max =
    {
        1,
        LayerCapabilities::InputElementCountMax,
        1,
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputUInt16Max;
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

static const RangeLimits<> & limitsForOutputUInt16Max()
{
    static const RangeLimits<> _limitsForOutputUInt16Max =
    {
        limitsForInputUInt16Max(),
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutputUInt16Max;
}

static const RangeLimits<> & limitsForInputShapeLegacy()
{
    static const RangeLimits<> _limitsForInputShapeLegacy =
    {
        LayerCapabilities::InputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementCountMultipliers(),
        Gna2StatusXnnErrorInputVolume
    };
    return _limitsForInputShapeLegacy;
}

static const MultiplierLimits & shapeLimitMultipliersForCnnLegacy()
{
    static const MultiplierLimits _shapeLimitMultipliersForCnnLegacy =
    {
        {{Gna2DataTypeInt8, 2 * XNN_N_IN_ELEMS_MPLY},
            {Gna2DataTypeInt16, XNN_N_IN_ELEMS_MPLY }},
            Gna2StatusCnnErrorConvFltVolume
    };
    return _shapeLimitMultipliersForCnnLegacy;
}
//
//static const MultiplierLimits & shapeLimitMultipliersFor1D()
//{
//    static const MultiplierLimits _shapeLimitMultipliersFor1D =
//    {
//        {{Gna2DataTypeInt8, 2 * 8},
//            {Gna2DataTypeInt16, 8 }},
//            Gna2StatusCnnErrorConvFltVolume
//    };
//    return _shapeLimitMultipliersFor1D;
//}

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
const FullCapabilitiesMap & ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},    // N = 1
                    {{GNA_DIM_H, limitsForInputEqual1()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_0_9)})},
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},    // N = 1
                    {{GNA_DIM_N, limitsForInputEqual1()},
                    {GNA_DIM_H, limitsForInputShapeLegacy()},
                    {GNA_DIM_W, limitsForInputEqual1()},
                    {GNA_DIM_D, limitsForInputEqual1()}},
                    GetModes(InputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},    // N = 1
                    {{GNA_DIM_N, limitsForInputEqual1()},
                    {GNA_DIM_H, limitsForInputUInt16Max()},
                    {GNA_DIM_W, limitsForInputUInt16Max()},
                    {GNA_DIM_D, limitsForInputUInt16Max()}},
                    GetModes(InputOperandIndex, GNA_3_0)})}
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
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
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                { GNA_TENSOR_HW },
                {{GNA_DIM_H, limitsForOutputEqual1()},
                {GNA_DIM_W, limitsForOutputUInt16Max()}},
                GetModes(OutputOperandIndex, GNA_0_9)})},
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_NHWD },
                    {{GNA_DIM_N, limitsForOutputEqual1()},
                    {GNA_DIM_H, {Filter1DElementsMultiplier, Filter1DElementsMax, Filter1DElementsMultiplier, Gna2StatusXnnErrorOutputVolume}},
                    {GNA_DIM_W, limitsForOutputUInt16Max()},
                    {GNA_DIM_D, limitsForOutputEqual1()}},
                    GetModes(OutputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},
                    {{GNA_DIM_N, limitsForOutputUInt16Max()},
                    {GNA_DIM_H, limitsForOutputUInt16Max()},
                    {GNA_DIM_W, limitsForOutputUInt16Max()},
                    {GNA_DIM_D, {1,
                        Filter2DCountMax /* bigger limit to workaround lack of 1D/2D differentiation */,
                        1, Gna2StatusXnnErrorOutputVolume}}},
                    GetModes(OutputOperandIndex, GNA_3_0)})}
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},
                    {{GNA_DIM_N, limitsForOutputEqual1()},
                    {GNA_DIM_H, limitsForOutputEqual1()},
                    {GNA_DIM_W, limitsForOutputUInt16Max()},
                    {GNA_DIM_D, {1, CNN_1D_N_KERNELS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
                    {{GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes}})}
            }},
        }},
        {FilterOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NW},    // N - # filters, W - # filter coefficients
                    {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusCnnErrorConvFltCount}},
                    {GNA_DIM_W, {CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, shapeLimitMultipliersForCnnLegacy(), Gna2StatusCnnErrorConvFltVolume}}},
                    {{ GNA_INT8, GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})}
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {1, Filter2DCountMax, 1, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, 2048, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR }, Gna2StatusXnnErrorConvFltBytes }})}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHWD},    // N - # filters, H - # filter coefficients
                    {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, shapeLimitMultipliersForCnnLegacy(), Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {1, 1, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {1, 1, 1, Gna2StatusCnnErrorConvFltVolume}}},
                    {{ GNA_INT8, GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
                    {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_1D_N_KERNELS_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusCnnErrorConvFltCount}},
                        {GNA_DIM_H, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, 1, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_W, {CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, CNN_N_FLT_COEFF_MIN, Gna2StatusCnnErrorConvFltVolume}},
                        {GNA_DIM_D, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, 1, Gna2StatusCnnErrorConvFltVolume}}},
                        // Padding to 16B is required for each Kernel
                    {{ GNA_INT16 }, Gna2StatusXnnErrorConvFltBytes }})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_CONVOLUTIONAL, {
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_N},          // H - #kernel (GNA_BIAS_PER_KERNEL)
                    {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen0_9()})},
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHW},          // N - #kernel (GNA_BIAS_PER_KERNEL)
                    {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_W, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                        {{GNA_DIM_N, {1, CNN_N_FLT_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen3Cnn2D()})}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHW},          // N - #kernel (GNA_BIAS_PER_KERNEL)
                    {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_W, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}}},
                    _ModesGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                        {{GNA_DIM_N, {1, CNN_N_FLT_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_H, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}},
                        {GNA_DIM_W, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}}},
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
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltStride}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, CNN_N_FLT_COEFF_MAX, 1, Gna2StatusCnnErrorConvFltStride}}}))}
            }},
        }},
        {ZeroPaddingParamIndex,{
            {INTEL_CONVOLUTIONAL_2D,{
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                        {{GNA_DIM_H, {0, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltPadding}},
                        {GNA_DIM_W, {0, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltPadding}}}))},
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {0, 0, 1, Gna2StatusCnnErrorConvFltPadding}},
                    {GNA_DIM_W, {0, 0, 1, Gna2StatusCnnErrorConvFltPadding}}}))}
            }},
        }},
        {PoolingStrideParamIndex,{
            {INTEL_CONVOLUTIONAL_2D, {
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorPoolStride}},
                    {GNA_DIM_W, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorPoolStride}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorPoolStride}},
                    {GNA_DIM_W, {1, CNN_POOL_SIZE_MAX, 1, Gna2StatusCnnErrorPoolStride}}}))}
            }},
        }},
        {PoolingWindowParamIndex,{
            {INTEL_CONVOLUTIONAL_2D, {
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorPoolSize}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {0, CNN_POOL_SIZE_MAX, 1, Gna2StatusCnnErrorPoolSize}}}))}
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
