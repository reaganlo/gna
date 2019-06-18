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

#include "ConvolutionalFunctions2D.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "KernelArguments.h"
#include "OperationConfig.h"
#include "Shape.h"
#include "Tensor.h"
#include "Validator.h"
#include "Transform.h"

#include "gna-api.h"

#include <map>
#include <memory>
#include <utility>

using namespace GNA;

const FullCapabilitiesMap ConvolutionFunction2D::strideLimits
{
    { INTEL_CONVOLUTIONAL_2D, {
        { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_WH},
            {{GNA_DIM_W, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltStride}},
             {GNA_DIM_H, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltStride}}}))}
    }}
};

const FullCapabilitiesMap ConvolutionFunction2D::paddingLimits
{
    { INTEL_CONVOLUTIONAL_2D, {
        { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_WH},
            {{GNA_DIM_W, {0, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltPadding}},
             {GNA_DIM_H, {0, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorConvFltPadding}}}))}
    }}
};

const FullCapabilitiesMap ConvolutionFunction2D::outputCapabilities
{
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes }})}
    }},
};

std::unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::Create(
    const TransformFactoryConfig& config, const OperationConfig& operationConfig)

{
    switch (config.validator.Operation)
    {
    case INTEL_CONVOLUTIONAL_2D:
    {
        return create(config, operationConfig);
    }
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

std::unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::create(
    const TransformFactoryConfig& config, const OperationConfig& operation)
{
    // TODO:3: P2: simplify to auto FiltersTensor = config.GetOperand(OperandTypeFilter);
    // TODO:3: P2: convert api types to internal so that transforms rely only on internal data types
    const auto filtersTensor = operation.FiltersTensor;
    auto filters = FiltersTensor::Create(filtersTensor, config.validator);

    auto stride = std::make_unique<const Component>(operation.ConvolutionStride,
        Validator{ config.validator, strideLimits });
    auto paddingShape = operation.ZeroPadding;
    if(paddingShape.empty())
    {
        paddingShape = Shape{ GNA_TENSOR_WH, 0u, 0u };
    }
    auto padding = std::make_unique<const Component>(paddingShape,
        Validator{ config.validator, paddingLimits });

    const Shape outputDims = GetOutputShape(config.input->Dimensions,
        filters->Dimensions, stride->Dimensions, padding->Dimensions);

    const auto biasTensor = operation.BiasesTensor;
    const auto biasMode = operation.BiasMode;

    auto biases = CreateBiasTensor(biasTensor, biasMode, filters->Count,
        outputDims, config.validator);

    return std::make_unique<ConvolutionFunction2D>(BaseTransformConfig<ConvolutionKernel2D>{config,
        AccelerationDetector::GetKernelMap<ConvolutionKernel2D>(
            KERNEL_CONVOLUTIONAL_2D,  { config.input->Mode, filters->Mode, (biases ? static_cast<gna_data_mode>(biases->Mode): GNA_DATA_DISABLED) })},
        move(filters), move(biases), move(stride), move(padding));
}

Shape ConvolutionFunction2D::CalculateBiasShape(const Gna2BiasMode mode, const uint32_t filterCount, Shape const & outputShape)
{
    switch (mode)
    {
    case Gna2BiasModeDefault:
    {
        return Shape(GNA_TENSOR_NHW, filterCount, 1u, 1u);
    }
    case Gna2BiasModePerStride:
    {
        return Shape(GNA_TENSOR_NHW,
            filterCount,
            outputShape.at(GNA_DIM_H),
            outputShape.at(GNA_DIM_W));
    }
    default:
    {
        return Shape(GNA_TENSOR_NHW, 1u, 1u, 1u); //TODO: FIX. Workaround for shape when bias_disabled
        //return Shape{};
    }
    }
}

std::unique_ptr<const BiasTensor> ConvolutionFunction2D::CreateBiasTensor(
    Gna2Tensor const & apiTensor, Gna2BiasMode biasMode, uint32_t filtersCount,
    Shape const & outputShape, const LayerValidator& validatorIn)
{
    Shape biasDims = CalculateBiasShape(biasMode, filtersCount, outputShape);
    // TODO:3: assert calculated bias shape matches one in apiTensor if provided by user (API2)
    return std::make_unique<const BiasTensor>(
        biasDims,
        0,
        DataMode{ apiTensor.Type, apiTensor.Mode },
        apiTensor.Data,
        validatorIn,
        biasMode);
}

Shape ConvolutionFunction2D::GetOutputShape(Shape const & inputShape,
        Shape const & filerShape, Shape const & strideShape, Shape const & paddingShape)
{
    Shape outputShape;
    outputShape.LayoutOrder = GNA_TENSOR_NHWD;
    outputShape[GNA_DIM_N] = inputShape.at(GNA_DIM_N);
    // save #filters as Depth dimension of output (D in filters is used for 3D convolution)
    outputShape[GNA_DIM_D] = filerShape.at(GNA_DIM_N);

    for (const auto& dimPair : strideShape)
    {
        auto const dim = dimPair.first;
        outputShape[dim] =
            1 + (inputShape.at(dim) + (2 * paddingShape.at(dim)) - filerShape.at(dim))
            / dimPair.second;
    }
    return outputShape;
}

ConvolutionFunction2D::ConvolutionFunction2D(const BaseTransformConfig<ConvolutionKernel2D>& config,
    std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<const Component> stride,
    std::unique_ptr<const Component> padding) :
    Transform{ConvolutionalTransform2D, &config.kernels, config.input},
    Biases{ move(biases) },
    Filters{ move(filters) },
    Stride{ move(stride) },
    Padding{ move(padding) }
{
    if (KernelBiasModePerFilter == Biases->BiasMode)
    {
        Expect::Equal<uint32_t>(Biases->at(GNA_DIM_H), 1, Gna2StatusXnnErrorBiasMode);
        Expect::Equal<uint32_t>(Biases->at(GNA_DIM_W), 1, Gna2StatusXnnErrorBiasMode);
    }

    Shape outputDims = GetOutputShape(Input->Dimensions, Filters->Dimensions,
        Stride->Dimensions, Padding->Dimensions);

    Output = std::make_unique<Tensor>(outputDims, DataMode{GNA_INT32}, config.outputBuffer,
        Validator{config.validator, outputCapabilities});

    auto out = Output->Dimensions;
    out.erase(GNA_DIM_D);
    //Expect::Fits(out, Input->Dimensions); //TODO: Check if this check is valid/needed

    gna_3d_dimensions input = Input->Dimensions;
    gna_3d_dimensions filter = Filters->Dimensions;
    gna_3d_dimensions convolutionStride = Stride->Dimensions;
    gna_3d_dimensions zeroPadding = Padding->Dimensions;

    auto kernelBiasMode = Biases->BiasMode;

    ConvolutionConfig2D kernelConvolutionConfig2D{ input.width, input.height, input.depth,Filters->at(GNA_DIM_N),
        filter.width, filter.height, filter.depth,
        KernelDataMode{Filters->Mode.Size}, Filters->Buffer,
        convolutionStride.width, convolutionStride.height,
        zeroPadding.width, zeroPadding.height,
        kernelBiasMode,
        KernelDataMode{Biases->Mode.Size},
        Biases->Buffer };

    hiddenConfig = std::make_unique<KernelConfig<ConvolutionConfig2D>>(
        kernelConvolutionConfig2D,
        BaseConfig{Input->Buffer, Output->Buffer});
}

Tensor const & ConvolutionFunction2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case 2:
    {
        return GetOperandIfExistOrThrow(Filters);
    }
    case 3:
    {
        return GetOperandIfExistOrThrow(Biases);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}