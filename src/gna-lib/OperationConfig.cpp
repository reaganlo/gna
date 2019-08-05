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

#include "OperationConfig.h"

#include "AccelerationDetector.h"
#include "Expect.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"
#include "common.h"

using namespace GNA;

namespace GNA
{

template<>
Gna2Tensor OperationConfig::getBiasTensor<intel_affine_multibias_func_t>(
    const nn_layer& layer, const intel_affine_multibias_func_t& affineFunc)
{
    Gna2Tensor a{};
    ModelWrapper::SetLayout(a, "HW");
    a.Type = DataMode(affineFunc.nBytesPerBias).Type;
    a.Data = affineFunc.pBiases;
    a.Mode = Gna2TensorModeDefault;
    a.Shape = { 2, layer.nOutputRows, affineFunc.biasVectorCount };

    return a;
}

}

OperationConfig::OperationConfig(const nn_layer& layer)
{
    InitOperationConfig(layer);
}

OperationConfig::OperationConfig(const Gna2Operation& apiOperation)
{
    InitOperationConfig(apiOperation);
}

Gna2OperationType OperationConfig::GetOperationType(const Gna2Operation& apiOperation)
{
    return apiOperation.Type;
}

Gna2OperationType OperationConfig::GetOperationType(const nn_layer& layer)
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
        return Gna2OperationTypeFullyConnectedAffine;
    case INTEL_AFFINE_DIAGONAL:
        return Gna2OperationTypeElementWiseAffine;
    case INTEL_CONVOLUTIONAL:
    case INTEL_CONVOLUTIONAL_2D:
        return Gna2OperationTypeConvolution;
    case INTEL_COPY:
        return Gna2OperationTypeCopy;
    case INTEL_DEINTERLEAVE:
    case INTEL_INTERLEAVE:
        return Gna2OperationTypeTransposition;
    case INTEL_GMM:
        return Gna2OperationTypeGmm;
    case INTEL_RECURRENT:
        return Gna2OperationTypeRecurrent;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
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
    switch(OperationType)
    {
    case Gna2OperationTypeConvolution:
        return ConvolutionalTransform2D;
    case Gna2OperationTypeCopy:
        return CopyTransform;
    case Gna2OperationTypeElementWiseAffine:
        return AffineDiagonalTransform;
    case Gna2OperationTypeFullyConnectedAffine:
        if (HasGroupedBias(BiasesTensor, BiasMode))
        {
            return AffineMultibiasTransform;
        }
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

bool OperationConfig::IsOperandAvailable(const Gna2Operation & operation, uint32_t index)
{
    return nullptr != operation.Operands &&
        index < operation.NumberOfOperands &&
        nullptr != operation.Operands[index];
}

bool OperationConfig::IsParameterAvailable(const Gna2Operation & operation, uint32_t index)
{
    return nullptr != operation.Parameters &&
        index < operation.NumberOfParameters &&
        nullptr != operation.Parameters[index];
}

bool OperationConfig::IsCNN1D(const Gna2Operation & operation)
{
    return Gna2OperationTypeConvolution == operation.Type &&
        2 == operation.Operands[InputOperandIndex]->Shape.NumberOfDimensions;
}

const nn_layer_cnn2d * OperationConfig::CastToCnn2DDetails(const nn_layer& layer)
{
    switch (layer.operation)
    {
    case INTEL_CONVOLUTIONAL_2D:
    {
        return static_cast<const nn_layer_cnn2d*>(layer.pLayerStruct);
    }
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

Gna2Tensor OperationConfig::GetWeights(const nn_layer& layer)
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
        return getWeightsTensor(layer,
            static_cast<const nn_layer_affine *>(layer.pLayerStruct)->affine);
    case INTEL_AFFINE_MULTIBIAS:
        return getWeightsTensor(layer,
            static_cast<const nn_layer_affine_multi *>(layer.pLayerStruct)->affine);
    case INTEL_RECURRENT:
        return getWeightsTensor(layer,
            static_cast<const nn_layer_recurrent *>(layer.pLayerStruct)->affine);
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

Gna2Tensor OperationConfig::GetWeights(const Gna2Operation & operation)
{
    const auto index = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexWeight);
    return GetOperand(operation, index, {});
}

Gna2Tensor OperationConfig::GetFilters(const nn_layer& layer)
{
    const auto cnn2d = CastToCnn2DDetails(layer);
    Gna2Tensor a{};
    a.Type = DataMode(cnn2d->convolution.filters.dataMode).Type;
    a.Shape = { 4, cnn2d->convolution.filters.count,
        cnn2d->convolution.filters.dimensions.height,
        cnn2d->convolution.filters.dimensions.width ,
        cnn2d->convolution.filters.dimensions.depth };
    a.Data = cnn2d->convolution.filters.filtersData;
    return a;
}

Gna2Tensor OperationConfig::GetFilters(const Gna2Operation & operation)
{
    return GetOperand(operation, FilterOperandIndex, {});
}

Gna2Tensor OperationConfig::GetBiases(const nn_layer& layer)
{
    if (isCNN2D(layer))
    {
        const auto cnn2d = CastToCnn2DDetails(layer);
        const auto& b = cnn2d->convolution.biases;
        Gna2Tensor t{};
        t.Data = b.biasesData;
        const DataMode dataModeLoc{ b.dataMode };
        t.Type = dataModeLoc.Type;
        t.Mode = dataModeLoc.Mode;
        return t;
    }

    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
        return getBiasTensor(layer,
            static_cast<const nn_layer_affine *>(layer.pLayerStruct)->affine);
    case INTEL_AFFINE_MULTIBIAS:
        return getBiasTensor(layer,
            static_cast<const nn_layer_affine_multi *>(layer.pLayerStruct)->affine);
    case INTEL_RECURRENT:
        return getBiasTensor(layer,
            static_cast<const nn_layer_recurrent *>(layer.pLayerStruct)->affine);
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

Gna2BiasMode OperationConfig::GetBiasMode(const nn_layer& layer)
{
    static std::map<gna_bias_mode, Gna2BiasMode> biasModeMap{
        { GNA_BIAS_PER_KERNEL, Gna2BiasModeDefault },
        { GNA_BIAS_PER_STRIDE, Gna2BiasModePerStride },
        { GNA_BIAS_NOT_SUPPORTED, Gna2BiasModeDefault },
    };

    if (isCNN2D(layer))
    {
        const auto cnn2d = CastToCnn2DDetails(layer);
        return biasModeMap.at(cnn2d->convolution.biases.mode);
    }

    if (layer.operation == INTEL_AFFINE_MULTIBIAS)
    {
        return Gna2BiasModeGrouping;
    }

    return Gna2BiasModeDefault;
}

Gna2Tensor OperationConfig::GetOperand(const Gna2Operation & operation, uint32_t index, Gna2Tensor defaultValue)
{
    if (IsOperandAvailable(operation, index))
    {
        return *(operation.Operands[index]);
    }
    return defaultValue;
}

Gna2BiasMode OperationConfig::GetBiasMode(const Gna2Operation & operation)
{
    return GetParameterAs<Gna2BiasMode>(operation, ParameterIndexBiasMode, Gna2BiasModeDefault);
}

Gna2Tensor OperationConfig::GetBiases(const Gna2Operation & operation)
{
    Gna2Tensor disabled{};
    disabled.Mode = Gna2TensorModeDisabled;
    return GetOperand(operation, BiasOperandIndex, disabled);
}

Shape OperationConfig::GetStride(const Gna2Operation & operation)
{
    const auto parameter = TryGetParamShape(operation, ParameterIndexConvolutionStride);
    return parameter;
}

Shape OperationConfig::GetStride(const nn_layer& layer)
{
    const auto cnn = CastToCnn2DDetails(layer);
    return Shape{ cnn->convolution.stride };
}

Shape OperationConfig::GetZeroPadding(const Gna2Operation& operation)
{
    const auto parameter = TryGetParamShape(operation, ParameterIndexZeroPadding);
    return parameter;
}

Shape OperationConfig::GetZeroPadding(const nn_layer& layer)
{
    //TODO:3:P1 generalize
    const auto cnn = CastToCnn2DDetails(layer);
    return Shape{ cnn->convolution.zeroPadding };
}

nn_layer_pool2d OperationConfig::GetPoolingImpl(const nn_layer& layer)
{
    switch (layer.operation)
    {
    case INTEL_CONVOLUTIONAL_2D:
    {
        auto cnn = static_cast<const nn_layer_cnn2d*>(layer.pLayerStruct);
        auto pooling = nn_layer_pool2d{ cnn->inputDimensions, cnn->pooling };
        return pooling;
    }
    case GNA_LAYER_CNN_2D_POOLING:
    {
        auto pooling = static_cast<const nn_layer_pool2d*>(layer.pLayerStruct);
        return *pooling;
    }
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

Shape OperationConfig::GetPoolingStride(const nn_layer_pool2d & pooling)
{
    return Shape{ pooling.pooling.stride };
}

Shape OperationConfig::GetPoolingWindow(const nn_layer_pool2d & pooling)
{
    return Shape{ pooling.pooling.window };
}

Shape OperationConfig::TryGetParamShape(const Gna2Operation & operation, OperationInfoKey parameter)
{
    auto const parameterIndex = ModelWrapper::GetOperationInfo(operation.Type, parameter);
    return TryGetParamShape(operation, parameterIndex);
}

Shape OperationConfig::TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex)
{
    //TODO:3:P2: Add if(IsRequired(operation, parameterIndex))
    const Gna2Shape shape = GetParameterAs<Gna2Shape>(operation, parameterIndex, {});
    return Shape::Create(shape, GNA_TENSOR_ORDER_ANY);
}

Shape OperationConfig::GetPoolingWindow(const Gna2Operation & operation)
{
    const auto parameter = TryGetParamShape(operation, ParameterIndexPoolingWindow);
    return parameter;
}

Shape OperationConfig::GetPoolingStride(const Gna2Operation & operation)
{
    const auto parameter = TryGetParamShape(operation, ParameterIndexPoolingStride);
    return parameter;
}

Gna2PoolingMode OperationConfig::GetPoolingMode(const nn_layer_pool2d & pooling)
{
    static std::map< intel_pool_type_t, Gna2PoolingMode> poolingModeMap{
        { INTEL_NO_POOLING, Gna2PoolingModeDisabled },
        { INTEL_MAX_POOLING, Gna2PoolingModeMax },
        { INTEL_SUM_POOLING, Gna2PoolingModeSum }
    };
    return poolingModeMap.at(pooling.pooling.type);
}

Gna2PoolingMode OperationConfig::GetPoolingMode(const Gna2Operation & operation)
{
    return GetParameterAs<Gna2PoolingMode>(operation, ParameterIndexPoolingMode, Gna2PoolingModeDisabled);
}

void OperationConfig::InitMultibias(const Gna2Operation& operation)
{
    auto bmIndex = ModelWrapper::GetOperationInfo(operation.Type, ParameterIndexBiasMode);
    BiasMode = *static_cast<Gna2BiasMode *>(operation.Parameters[bmIndex]);

    auto bviIndex = ModelWrapper::GetOperationInfo(
            operation.Type, ParameterIndexBiasVectorIndex);
    BiasVectorIndex = *static_cast<uint32_t *>(operation.Parameters[bviIndex]);

    auto wsfIndex = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexWeightScaleFactors);

    if (operation.Operands[wsfIndex] != nullptr)
    {
        WeightScalesTensor = *operation.Operands[wsfIndex];
    }
}

void OperationConfig::InitMultibias(const nn_layer& layer)
{
    auto affineMulti = static_cast<const nn_layer_affine_multi *>(layer.pLayerStruct);
    auto affineMultiFunc = &affineMulti->affine;
    BiasVectorIndex = affineMultiFunc->biasVectorIndex;
    BiasMode = Gna2BiasModeGrouping;

    if (affineMultiFunc->nBytesPerWeight == 1)
    {
        WeightScalesTensor.Data = affineMultiFunc->weightScaleFactors;
        WeightScalesTensor.Type = Gna2DataTypeWeightScaleFactor;
        WeightScalesTensor.Mode = Gna2TensorModeDefault;
        WeightScalesTensor.Shape = { 1, layer.nOutputRows };
        ModelWrapper::SetLayout(WeightScalesTensor, "H");
    }
}

void OperationConfig::InitPooling(const Gna2Operation & operation)
{
    PoolingWindow = GetPoolingWindow(operation);
    PoolingStride = GetPoolingStride(operation);
    Mode = GetPoolingMode(operation);
}

void OperationConfig::InitPooling(const nn_layer& layer)
{
    const auto p = GetPoolingImpl(layer);
    PoolingWindow = GetPoolingWindow(p);
    PoolingStride = GetPoolingStride(p);
    Mode = GetPoolingMode(p);
}

bool OperationConfig::HasGroupedBias(
    const Gna2Tensor& biasTensor, const Gna2BiasMode biasMode)
{
    return biasTensor.Mode == Gna2TensorModeDefault
        && biasTensor.Shape.NumberOfDimensions == 2
        && biasMode == Gna2BiasModeGrouping;
}

bool OperationConfig::HasGroupedBias() const
{
    return HasGroupedBias(BiasesTensor, BiasMode);
}

bool OperationConfig::hasPooling(const Gna2Operation & operation)
{
    const auto indexPoolingMode = ModelWrapper::GetOperationInfo(operation.Type,
        ParameterIndexPoolingMode);
    const auto indexPoolingStride = ModelWrapper::GetOperationInfo(operation.Type,
        ParameterIndexPoolingStride);
    const auto indexPoolingWindow = ModelWrapper::GetOperationInfo(operation.Type,
        ParameterIndexPoolingWindow);
    return IsParameterAvailable(operation, indexPoolingMode) &&
        IsParameterAvailable(operation, indexPoolingStride) &&
        IsParameterAvailable(operation, indexPoolingWindow);
}

bool OperationConfig::hasPooling(const nn_layer& layer)
{
    return (layer.operation == INTEL_CONVOLUTIONAL_2D ||
        layer.operation == GNA_LAYER_CNN_2D_POOLING);
}

bool OperationConfig::isAffine(const nn_layer& layer)
{
    return layer.operation == INTEL_AFFINE
        || layer.operation == INTEL_AFFINE_DIAGONAL
        || layer.operation == INTEL_AFFINE_MULTIBIAS
        || layer.operation == INTEL_RECURRENT;
}

bool OperationConfig::isAffine(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeFullyConnectedAffine
        || operation.Type == Gna2OperationTypeElementWiseAffine
        || operation.Type == Gna2OperationTypeRecurrent;
}

bool OperationConfig::IsMultibias(const nn_layer& layer)
{
    return layer.operation == INTEL_AFFINE_MULTIBIAS;
}

bool OperationConfig::IsMultibias(const Gna2Operation & operation)
{
    if (operation.Type != Gna2OperationTypeFullyConnectedAffine)
    {
        return false;
    }

    auto biasModeIndex = ModelWrapper::GetOperationInfo(operation.Type, ParameterIndexBiasMode);
    if (operation.Parameters[biasModeIndex] == nullptr)
    {
        return false;
    }

    auto biasMode = *static_cast<Gna2BiasMode *>(operation.Parameters[biasModeIndex]);
    auto biasIndex = ModelWrapper::GetOperationInfo(operation.Type, OperandIndexBias);
    auto biasTensor = *operation.Operands[biasIndex];
    return HasGroupedBias(biasTensor, biasMode);
}

bool OperationConfig::isCNN2D(const nn_layer& layer)
{
    return INTEL_CONVOLUTIONAL_2D == layer.operation;
}

bool OperationConfig::isCNN2D(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeConvolution;
}

bool OperationConfig::isRecurrent(const nn_layer& layer)
{
    return layer.operation == INTEL_RECURRENT;
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

uint32_t OperationConfig::GetFeedbackDelay(const nn_layer& layer)
{
    auto rnnLayer = reinterpret_cast<intel_recurrent_layer_t *>(layer.pLayerStruct);
    return rnnLayer->feedbackFrameDelay;
}

