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

#include "OperationConfig.h"

#include "AccelerationDetector.h"
#include "Expect.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"

using namespace GNA;

OperationConfig::OperationConfig(const Gna2Operation& apiOperation) :
    Operation{ &apiOperation }
{
    InitOperationConfig(apiOperation);
}

Gna2OperationType OperationConfig::GetOperationType(const Gna2Operation& apiOperation)
{
    return apiOperation.Type;
}

kernel_op OperationConfig::GetKernelOperation() const
{
    switch (OperationType)
    {
    case Gna2OperationTypeConvolution:
        return KERNEL_CONVOLUTIONAL_2D;
    case Gna2OperationTypeCopy:
        return KERNEL_COPY;
    case Gna2OperationTypeElementWiseAffine:
        return KERNEL_AFFINE_DIAGONAL;
    case Gna2OperationTypeFullyConnectedAffine:
    case Gna2OperationTypeThreshold:        // TODO: 3: Introduce separate type
        return KERNEL_AFFINE;
    case Gna2OperationTypeGmm:
        return KERNEL_GMM;
    case Gna2OperationTypeRecurrent:
        return KERNEL_RECURRENT;
    case Gna2OperationTypeTransposition:
        return KERNEL_TRANSPOSE;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

TransformOperation OperationConfig::GetTransformOperation() const
{
    switch (OperationType)
    {
    case Gna2OperationTypeConvolution:
        return ConvolutionalTransform2D;
    case Gna2OperationTypeCopy:
        return CopyTransform;
    case Gna2OperationTypeElementWiseAffine:
        return AffineDiagonalTransform;
    case Gna2OperationTypeFullyConnectedAffine:
        if (BiasMode == Gna2BiasModeGrouping)
        {
            return AffineMultibiasTransform;
        }
        return AffineTransform;
    case Gna2OperationTypeThreshold:    // TODO: 3: Introduce separate type
        return AffineTransform;
    case Gna2OperationTypeGmm:
        return GmmTransform;
    case Gna2OperationTypeRecurrent:
        return RecurrentTransform;
    case Gna2OperationTypeTransposition:
        return TransposeTransform;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

bool OperationConfig::IsCNN1D(const Gna2Operation & operation)
{
    return Gna2OperationTypeConvolution == operation.Type &&
        2 == operation.Operands[InputOperandIndex]->Shape.NumberOfDimensions;
}

Gna2Tensor OperationConfig::GetWeights(const Gna2Operation & operation)
{
    const auto index = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexWeight);
    return ModelWrapper::GetOperand(operation, index, {});
}

Gna2Tensor OperationConfig::GetFilters(const Gna2Operation & operation)
{
    auto filter = ModelWrapper::GetOperand(operation, FilterOperandIndex, {});
    if (2 == filter.Shape.NumberOfDimensions)
    {
        filter.Shape.NumberOfDimensions = 4;
        filter.Shape.Dimensions[2] = filter.Shape.Dimensions[1];
        filter.Shape.Dimensions[1] = 1;
        filter.Shape.Dimensions[3] = 1;
    }
    return filter;
}

Gna2BiasMode OperationConfig::GetBiasMode(const Gna2Operation & operation)
{
    const auto biasMode = ModelWrapper::GetOptionalParameter<Gna2BiasMode>(operation, BiasModeConvolutionParamIndex, Gna2BiasModeDefault);
    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectInSet(biasMode, { Gna2BiasModeDefault, Gna2BiasModePerStride, Gna2BiasModePerStride }, Gna2ItemTypeParameter);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, static_cast<int32_t>(BiasModeConvolutionParamIndex));
    return biasMode;
}

Gna2Tensor OperationConfig::GetBiases(const Gna2Operation & operation)
{
    if (operation.Type == Gna2OperationTypeGmm)
    {
        return ModelWrapper::GetDisabledOperand();
    }
    return ModelWrapper::GetOperand(operation, BiasOperandIndex, ModelWrapper::GetDisabledOperand());
}

Shape OperationConfig::GetStride(const Gna2Operation & operation)
{
    auto parameter = GetShapeParameterOfMaximal2Dimensions(operation, ConvolutionStrideParamIndex);
    if (parameter.size() == 1)
    {
        parameter.LayoutOrder = Layout("HW");
        parameter['W'] = parameter['N'];
        parameter.erase(GNA_DIM_N);
        parameter['H'] = 1;
    }
    return parameter;
}

Shape OperationConfig::GetZeroPadding(const Gna2Operation& operation)
{
    auto parameter = GetShapeParameterOfMaximal2Dimensions(operation, ZeroPaddingParamIndex);
    if (parameter.size() == 1)
    {
        parameter = Shape();
    }
    return parameter;
}

Shape OperationConfig::TryGetParamShape(const Gna2Operation & operation, OperationInfoKey parameter)
{
    auto const parameterIndex = ModelWrapper::GetOperationInfo(operation.Type, parameter);
    return TryGetParamShape(operation, parameterIndex);
}

Shape OperationConfig::TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex)
{
    //TODO:3:P2: Add if(IsRequired(operation, parameterIndex))
    const Gna2Shape shape = ModelWrapper::GetOptionalParameter<Gna2Shape>(operation, parameterIndex, {});
    return Shape::Create(shape, GNA_TENSOR_ORDER_ANY);
}

Shape OperationConfig::GetShapeParameterOfMaximal2Dimensions(const Gna2Operation & operation, const uint32_t parameterIndex)
{
    const auto parameter = TryGetParamShape(operation, parameterIndex);
    const std::function<void()> command = [&]()
    {
        // TODO: 3: consider moving this check to Shape::Reshape,
        // when the validation happens for Component's ctor
        ModelErrorHelper::ExpectBelowEq(parameter.size(), 2, Gna2ItemTypeShapeNumberOfDimensions);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, static_cast<int32_t>(parameterIndex));
    return parameter;
}

Gna2PoolingMode OperationConfig::GetPoolingMode(const Gna2Operation & operation)
{
    return ModelWrapper::GetOptionalParameter<Gna2PoolingMode>(operation, ParameterIndexPoolingMode, Gna2PoolingModeDisabled);
}

void OperationConfig::InitMultibias(const Gna2Operation& operation)
{
    auto bmIndex = ModelWrapper::GetOperationInfo(operation.Type, ParameterIndexBiasMode);
    BiasMode = ModelWrapper::GetParameter<Gna2BiasMode>(operation, bmIndex);

    auto bviIndex = ModelWrapper::GetOperationInfo(
        operation.Type, ParameterIndexBiasVectorIndex);
    BiasVectorIndex = ModelWrapper::GetParameter<uint32_t>(operation, bviIndex);

    auto wsfIndex = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexWeightScaleFactors);

    // GNA 2.0 backward compatibility only
    if (Gna2DataTypeInt8 == WeightsTensor.Type
        && Gna2DataTypeInt16 == operation.Operands[InputOperandIndex]->Type)
    {
        WeightScalesTensor = ModelWrapper::GetEnabledOperand(operation, wsfIndex);
        ModelWrapper::SetLayout(WeightScalesTensor, "H");
    }
}

void OperationConfig::InitPooling(const Gna2Operation & operation)
{
    PoolingWindow = GetShapeParameterOfMaximal2Dimensions(operation, PoolingWindowParamIndex);
    if (PoolingWindow.size() == 1)
    {
        PoolingWindow.LayoutOrder = Layout("HW");
        PoolingWindow['W'] = PoolingWindow['N'];
        PoolingWindow.erase(GNA_DIM_N);
        PoolingWindow['H'] = 1;
    }
    PoolingStride = GetShapeParameterOfMaximal2Dimensions(operation, PoolingStrideParamIndex);
    if (PoolingStride.size() == 1)
    {
        PoolingStride.LayoutOrder = Layout("HW");
        PoolingStride['W'] = PoolingStride['N'];
        PoolingStride.erase(GNA_DIM_N);
        PoolingStride['H'] = 1;
    }
    Mode = GetPoolingMode(operation);
}

Gna2Tensor OperationConfig::GetEnabledOperand(uint32_t index) const
{
    Expect::NotNull(Operation);
    return ModelWrapper::GetEnabledOperand(*Operation, index);
}

bool OperationConfig::hasPooling(const Gna2Operation & operation)
{
    const auto poolingMode = ModelWrapper::GetOptionalParameter<Gna2PoolingMode>(operation, PoolingModeParamIndex, Gna2PoolingModeDisabled);
    if (poolingMode == Gna2PoolingModeDisabled)
    {
        ModelWrapper::ExpectParameterNotAvailable(operation, PoolingWindowParamIndex);
        ModelWrapper::ExpectParameterNotAvailable(operation, PoolingStrideParamIndex);
        return false;
    }
    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectInSet(poolingMode, { Gna2PoolingModeMax, Gna2PoolingModeSum }, Gna2ItemTypeParameter);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, static_cast<int32_t>(PoolingModeParamIndex));

    ModelWrapper::ExpectParameterAvailable(operation, PoolingWindowParamIndex);
    ModelWrapper::ExpectParameterAvailable(operation, PoolingStrideParamIndex);
    return true;
}

bool OperationConfig::isAffine(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeFullyConnectedAffine
        || operation.Type == Gna2OperationTypeElementWiseAffine
        || operation.Type == Gna2OperationTypeThreshold
        || operation.Type == Gna2OperationTypeRecurrent;
}

bool OperationConfig::IsMultibias(const Gna2Operation & operation)
{
    if (operation.Type != Gna2OperationTypeFullyConnectedAffine)
    {
        return false;
    }

    if (!ModelWrapper::HasParameter(operation, BiasModeAffineParamIndex))
    {
        return false;
    }
    const auto biasMode = *static_cast<Gna2BiasMode *>(operation.Parameters[BiasModeAffineParamIndex]);

    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectInSet(biasMode, { Gna2BiasModeDefault, Gna2BiasModeGrouping }, Gna2ItemTypeParameter);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, static_cast<int32_t>(BiasModeAffineParamIndex));

    return biasMode == Gna2BiasModeGrouping;
}

bool OperationConfig::isCNN2D(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeConvolution;
}

bool OperationConfig::isRecurrent(const Gna2Operation& operation)
{
    return operation.Type == Gna2OperationTypeRecurrent;
}

uint32_t OperationConfig::GetFeedbackDelay(const Gna2Operation& operation)
{
    auto delayIndex = ModelWrapper::GetOperationInfo(
        operation.Type, ParameterIndexDelay);
    return ModelWrapper::GetParameter<uint32_t>(operation, delayIndex);
}
