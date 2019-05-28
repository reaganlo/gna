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

using namespace GNA;


OperationConfig::OperationConfig(const nn_layer& layer)
{
    InitOperationConfig(layer);
}

OperationConfig::OperationConfig(const Gna2Operation& apiOperation)
{
    InitOperationConfig(apiOperation);
}

void OperationConfig::ExpectParameterAvailable(const Gna2Operation & operation, uint32_t index)
{
    Expect::NotNull(operation.Parameters, Gna2StatusXnnErrorLyrOperation);
    Expect::True(index < operation.NumberOfParameters, Gna2StatusXnnErrorLyrOperation);
    Expect::NotNull(operation.Parameters[index]);
}

bool OperationConfig::IsCNN1D(const Gna2Operation & operation)
{
    return Gna2OperationTypeConvolution == operation.Type && 2 == operation.Operands[0]->Shape.NumberOfDimensions;
}

const nn_layer_cnn2d * OperationConfig::GetNnLayerCnn2D_(const nn_layer& layer)
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
    const auto cnn2d = GetNnLayerCnn2D_(layer);
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
    if (operation.Type == Gna2OperationTypeConvolution &&
        operation.Operands[2] != nullptr)
    {
        return *operation.Operands[2];
    }
    //TODO:3:P1:Implement other cases if needed
    return Gna2Tensor{};
}

Gna2Tensor OperationConfig::GetBiases(const nn_layer& layer)
{
    const auto cnn2d = GetNnLayerCnn2D_(layer);
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
    const auto cnn2d = GetNnLayerCnn2D_(layer);
    return biasModeMap.at(cnn2d->convolution.biases.mode);
}

Gna2BiasMode OperationConfig::GetBiasMode(const Gna2Operation & operation)
{
    return GetParameterAs<Gna2BiasMode>(operation, 1);
}

Gna2Tensor OperationConfig::GetBiases(const Gna2Operation & operation)
{
    //TODO:3:P1 generalize
    const uint32_t biasIndex{ 3 };
    if (biasIndex < operation.NumberOfOperands && operation.Type != Gna2OperationTypeGmm)
    {
        if(nullptr != operation.Operands[biasIndex])
        return *operation.Operands[biasIndex];
    }
    Gna2Tensor out{};
    out.Mode = Gna2TensorModeDisabled;
    return out;
}

Shape OperationConfig::GetStrideWHD(const Gna2Operation & operation)
{
    const auto parameter = TryGetParamShape(operation, 0);
    return parameter;
}

Shape OperationConfig::GetStrideWHD(const nn_layer& layer)
{
    auto cnn = GetNnLayerCnn2D_(layer);
    return Shape{ cnn->convolution.stride };
}

Shape OperationConfig::GetZeroPadding(const Gna2Operation& operation)
{
    const auto parameter = TryGetParamShape(operation, 5);
    return parameter;
}

Shape OperationConfig::GetZeroPadding(const nn_layer& layer)
{
    //TODO:3:P1 generalize
    auto cnn = GetNnLayerCnn2D_(layer);
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
Shape OperationConfig::TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex)
{
    if (operation.NumberOfParameters > parameterIndex && operation.Parameters[parameterIndex] != nullptr)
    {
        const auto s = *static_cast<Gna2Shape*>(operation.Parameters[parameterIndex]);
        return Shape::Create(s, GNA_TENSOR_ORDER_ANY);
    }
    //TODO:3:P2: Add if(IsRequired(operation, parameterIndex))
    //{
    //    throw
    //}
    return {};
}

Shape OperationConfig::GetPoolingWindow(const Gna2Operation & operation)
{
    const auto parameter = TryGetParamShape(operation, 3);
    return parameter;
}

Shape OperationConfig::GetPoolingStride(const Gna2Operation & operation)
{
    const auto parameter = TryGetParamShape(operation, 4);
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
    const uint32_t parameterIndex = 2;
    Expect::Equal(operation.Type, Gna2OperationTypeConvolution, Gna2StatusXnnErrorLyrOperation);
    if (operation.NumberOfParameters > parameterIndex && operation.Parameters[parameterIndex] != nullptr)
    {
        const auto mode = *static_cast<Gna2PoolingMode*>(operation.Parameters[parameterIndex]);
        return mode;
    }
    return Gna2PoolingModeDisabled;
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
    return (operation.Parameters[2] != nullptr &&
        operation.Parameters[3] != nullptr &&
        operation.Parameters[4] != nullptr);
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
