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

#include "Expect.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"
#include "common.h"

using namespace GNA;

OperationConfig::OperationConfig(const nn_layer& layer)
{
    InitOperationConfig(layer);
}

OperationConfig::OperationConfig(const Gna2Operation& apiOperation)
{
    InitOperationConfig(apiOperation);
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
        2 == operation.Operands[0]->Shape.NumberOfDimensions;
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
    return GetOperand(operation, FilterComponent, {});
}

Gna2Tensor OperationConfig::GetBiases(const nn_layer& layer)
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

Gna2BiasMode OperationConfig::GetBiasMode(const nn_layer& layer)
{
    static std::map<gna_bias_mode, Gna2BiasMode> biasModeMap{
        { GNA_BIAS_PER_KERNEL, Gna2BiasModeDefault },
        { GNA_BIAS_PER_STRIDE, Gna2BiasModePerStride },
        { GNA_BIAS_NOT_SUPPORTED, Gna2BiasModeDefault },
    };
    const auto cnn2d = CastToCnn2DDetails(layer);
    return biasModeMap.at(cnn2d->convolution.biases.mode);
}

Gna2Tensor OperationConfig::GetOperand(const Gna2Operation & operation, GnaComponentType operand, Gna2Tensor defaultValue)
{
    auto const index = ModelWrapper::GetOperandIndex(operand);
    return  GetOperand(operation, index, defaultValue);
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
    return GetOperand(operation, BiasComponent, disabled);
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

bool OperationConfig::isCNN2D(const nn_layer& layer)
{
    return INTEL_CONVOLUTIONAL_2D == layer.operation;
}

bool OperationConfig::isCNN2D(const Gna2Operation & operation)
{
    return operation.Type == Gna2OperationTypeConvolution;
}
