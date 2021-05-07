/*
 INTEL CONFIDENTIAL
 Copyright 2019-2020 Intel Corporation.

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
#include "ConvolutionalLayer2D.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "KernelArguments.h"
#include "OperationConfig.h"
#include "Shape.h"
#include "Tensor.h"
#include "Validator.h"
#include "Transform.h"


#include <map>
#include <memory>
#include <utility>

using namespace GNA;

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
    auto filters = FiltersTensor::Create(operation.FiltersTensor,
            config.validator);

    auto stride = ConvolutionalLayer2D::CreateComponentFromParameter(operation.ConvolutionStride,
        config.validator, ConvolutionStrideParamIndex);

    auto padding = ConvolutionalLayer2D::CreateComponentFromParameter(operation.ZeroPadding,
        config.validator, ZeroPaddingParamIndex);

    const Shape outputDims = GetOutputShape(config.input->Dimensions,
        filters->Dimensions, stride->Dimensions, padding->Dimensions);

    const auto biasTensor = operation.BiasesTensor;
    const auto biasMode = operation.BiasMode;

    auto biases = CreateBiasTensor(biasTensor, biasMode, filters->Count,
        outputDims, config.validator);

    return std::make_unique<ConvolutionFunction2D>(BaseTransformConfig<ConvolutionKernel2D>{config,
        AccelerationDetector::GetKernelMap<ConvolutionKernel2D>(
            KERNEL_CONVOLUTIONAL_2D, { config.input->Mode, filters->Mode, (biases ? biases->Mode : DataMode{}) })},
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
    const std::function<std::unique_ptr<const BiasTensor>(const LayerValidator&)> buildWithValidator = [&](const auto& cnnValidator)
    {
        return std::make_unique<const BiasTensor>(
            biasDims,
            0,
            DataMode{ apiTensor.Type, apiTensor.Mode },
            apiTensor.Data,
            cnnValidator,
            biasMode);
    };
    try
    {
         // try new CNN using 1D variant
        auto const validator1D = LayerValidator{ validatorIn, INTEL_CONVOLUTIONAL_1D };
        return buildWithValidator(validator1D);
    }
    catch (const GnaException&)
    {
        return buildWithValidator(validatorIn);
    }
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

bool IsInput1D(const Shape& inputShape)
{
    try
    {
        return inputShape.at('N') == 1 && inputShape.at('H') == 1 && inputShape.at('D') == 1;
    }
    catch (...)
    {
        throw GnaException(Gna2StatusModelConfigurationInvalid);
    }
}

ConvolutionFunction2D::ConvolutionFunction2D(const BaseTransformConfig<ConvolutionKernel2D>& config,
    std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<const Component> stride,
    std::unique_ptr<const Component> padding) :
    Transform{ ConvolutionalTransform2D, &config.kernels, config.input },
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

    Expect::InRange(Filters->at(GNA_DIM_W), Input->at(GNA_DIM_W),
        Gna2StatusCnnErrorConvFltVolume);
    Expect::InRange(Filters->at(GNA_DIM_H), Input->at(GNA_DIM_H),
        Gna2StatusCnnErrorConvFltVolume);
    Expect::InRange(Stride->at(GNA_DIM_W), Filters->at(GNA_DIM_W),
        Gna2StatusCnnErrorConvFltVolume);
    Expect::InRange(Stride->at(GNA_DIM_H), Filters->at(GNA_DIM_H),
        Gna2StatusCnnErrorConvFltVolume);

    auto effectiveOperation = INTEL_CONVOLUTIONAL_2D;
    if (Gna2DeviceGeneration3_5 < config.validator.HwCapabilities.GetDeviceGeneration() &&
        INTEL_CONVOLUTIONAL_1D == Filters->GetEffectiveOperationType() &&
        INTEL_CONVOLUTIONAL_1D == Stride->GetEffectiveOperationType() &&
        IsInput1D(config.input->Dimensions))
    {
        is1D = true;
        effectiveOperation = INTEL_CONVOLUTIONAL_1D;
    }
    Expect::InRange(Padding->at(GNA_DIM_W), Filters->at(GNA_DIM_W) - 1,
        Gna2StatusCnnErrorConvFltPadding);
    Expect::InRange(Padding->at(GNA_DIM_H), Filters->at(GNA_DIM_H) - 1,
        Gna2StatusCnnErrorConvFltPadding);

    Shape outputDims = GetOutputShape(Input->Dimensions, Filters->Dimensions,
        Stride->Dimensions, Padding->Dimensions);

    auto const validatorOut = LayerValidator{ config.validator, effectiveOperation};
    Output = std::make_unique<Tensor>(outputDims, DataMode{ Gna2DataTypeInt32 }, config.outputBuffer,
        Validator{ validatorOut, ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex) });

    auto out = Output->Dimensions;
    out.erase(GNA_DIM_D);
    //Expect::Fits(out, Input->Dimensions); //TODO: Check if this check is valid/needed

    auto kernelBiasMode = Biases->BiasMode;

    ConvolutionConfig2D kernelConvolutionConfig2D{
        Input->at(GNA_DIM_W),
        Input->at(GNA_DIM_H),
        Input->at(GNA_DIM_D),
        Filters->at(GNA_DIM_N),
        Filters->at(GNA_DIM_W),
        Filters->at(GNA_DIM_H),
        Filters->at(GNA_DIM_D),
        KernelDataMode{Filters->Mode.Size}, Filters->Buffer,
        Stride->at(GNA_DIM_W),
        Stride->at(GNA_DIM_H),
        Padding->at(GNA_DIM_W),
        Padding->at(GNA_DIM_H),
        kernelBiasMode,
        KernelDataMode{Biases->Mode.Size},
        Biases->Buffer };

    hiddenConfig = std::make_unique<KernelConfig<ConvolutionConfig2D>>(
        kernelConvolutionConfig2D,
        BaseConfig{ Input->Buffer, Output->Buffer });
}

Tensor const & ConvolutionFunction2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case FilterOperandIndex:
    {
        return GetOperandIfExistOrThrow(Filters);
    }
    case BiasOperandIndex:
    {
        return GetOperandIfExistOrThrow(Biases);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}
