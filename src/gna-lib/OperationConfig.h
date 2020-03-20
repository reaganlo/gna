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

#pragma once

#include "AccelerationDetector.h"
#include "DataMode.h"
#include "Shape.h"
#include "ModelError.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"
#include "common.h"

namespace GNA
{
class OperationConfig
{
public:
    OperationConfig(const nn_layer& layer);

    OperationConfig(const Gna2Operation& apiOperation);

    static Gna2PoolingMode GetPoolingMode(const Gna2Operation& operation);
    static bool IsCNN1D(const Gna2Operation & operation);

    static bool IsMultibias(const nn_layer& layer);
    static bool IsMultibias(const Gna2Operation& operation);

    static Gna2OperationType GetOperationType(const Gna2Operation& operation);
    static Gna2OperationType GetOperationType(const nn_layer& layer);

    static bool HasGroupedBias(
        const Gna2Tensor& biasTensor, const Gna2BiasMode biasMode);

    kernel_op GetKernelOperation() const;
    TransformOperation GetTransformOperation() const;

    bool HasGroupedBias() const;

    Gna2Tensor GetEnabledOperand(uint32_t index) const;

    template<typename T>
    T GetParameterAs(uint32_t index) const
    {
        Expect::NotNull(Operation);
        if (IsParameterAvailable(*Operation, index))
        {
            return *static_cast<T*> (Operation->Parameters[index]);
        }
        throw GnaException(Gna2StatusModelConfigurationInvalid);
    }

    template<typename Target, typename Source>
    static std::unique_ptr<const Target> CreateCnnTarget(
        const Source& source, const LayerValidator& validator, const FullCapabilitiesMap& caps)
    {
        try
        {
            // 1D CNN in new arch
            auto const validator1D = LayerValidator{ validator, INTEL_CONVOLUTIONAL_1D };
            return std::make_unique<const Target>(source,
                Validator{ validator1D, caps });
        }
        catch (const GnaException&)
        {
            // try 2D CNN in new arch
            return std::make_unique<const Target>(source,
                Validator{ validator, caps });
        }
    }

    static std::unique_ptr<const Component> CreateCnnComponent(const Shape& shape,
        const LayerValidator& validator, const FullCapabilitiesMap & caps)
    {
        if (shape.empty())
        {
            return CreateCnnTarget<Component, Shape>(
                Shape{ GNA_TENSOR_HW, 0u, 0u }, validator, caps);
        }
        else
        {
            return CreateCnnTarget<Component, Shape>(shape, validator, caps);
        }
    }

    Gna2OperationType OperationType;
    Gna2Tensor WeightsTensor;
    Gna2Tensor FiltersTensor;
    Shape ConvolutionStride;
    Shape ZeroPadding;
    Shape PoolingWindow;
    Shape PoolingStride;
    Gna2PoolingMode Mode;
    Gna2Tensor BiasesTensor;
    Gna2BiasMode BiasMode = Gna2BiasModeDefault;
    uint32_t FeedbackDelay;

    Gna2Tensor WeightScalesTensor;
    uint32_t BiasVectorIndex;

    Gna2Operation const * const Operation;

protected:
    template<class T>
    void InitOperationConfig(const T& operation)
    {
        OperationType = GetOperationType(operation);
        BiasesTensor = GetBiases(operation);
        // TODO: 3: Remove when full (e.g., bias) buffer addition and late sanity checking implemented
        if (BiasesTensor.Mode != Gna2TensorModeDisabled)
        {
            ModelErrorHelper::ExpectNotNull(BiasesTensor.Data, Gna2ItemTypeOperandData, BiasOperandIndex);
        }

        if (isAffine(operation))
        {
            WeightsTensor = GetWeights(operation);
            // TODO: 3: Remove when full (e.g., weights) buffer addition and late sanity checking implemented
            ModelErrorHelper::ExpectNotNull(WeightsTensor.Data, Gna2ItemTypeOperandData, WeightOperandIndex);

            if (IsMultibias(operation))
            {
                InitMultibias(operation);
            }
        }
        if (isRecurrent(operation))
        {
            FeedbackDelay = GetFeedbackDelay(operation);
        }
        if (isCNN2D(operation))
        {
            FiltersTensor = GetFilters(operation);
            // TODO: 3: Remove when full (e.g., filers) buffer addition and late sanity checking implemented
            ModelErrorHelper::ExpectNotNull(FiltersTensor.Data, Gna2ItemTypeOperandData, FilterOperandIndex);
            ConvolutionStride = GetStride(operation);
            ZeroPadding = GetZeroPadding(operation);
            BiasMode = GetBiasMode(operation);
            if (hasPooling(operation))
            {
                InitPooling(operation);
            }
            else
            {
                Mode = Gna2PoolingModeDisabled;
            }
        }
    }
    void InitPooling(const Gna2Operation& operation);
    void InitPooling(const nn_layer& layer);

    void InitMultibias(const Gna2Operation& operation);
    void InitMultibias(const nn_layer& layer);

private:
    static Shape TryGetParamShape(const Gna2Operation & operation, OperationInfoKey parameter);
    static Shape TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex);

    static const nn_layer_cnn2d* CastToCnn2DDetails(const nn_layer& layer);

    static Gna2Tensor GetWeights(const nn_layer& layer);
    static Gna2Tensor GetWeights(const Gna2Operation& operation);
    static Gna2Tensor GetFilters(const nn_layer& layer);
    static Gna2Tensor GetFilters(const Gna2Operation& operation);

    static bool IsOperandAvailable(const Gna2Operation & operation, uint32_t index);
    static bool IsParameterAvailable(const Gna2Operation & operation, uint32_t index);

    template<typename T>
    static T GetParameterAs(const Gna2Operation & operation, OperationInfoKey parameter, T defaultValue)
    {
        auto const parameterIndex = ModelWrapper::GetOperationInfo(operation.Type, parameter);
        return GetParameterAs(operation, parameterIndex, defaultValue);
    }

    template<typename T>
    static Gna2Tensor getWeightsTensor(const nn_layer& layer, const T& affineFunc)
    {
        Gna2Tensor a{};
        a.Type = DataMode(affineFunc.nBytesPerWeight).Type;
        a.Data = affineFunc.pWeights;
        a.Mode = Gna2TensorModeDefault;
        if (layer.operation == INTEL_AFFINE_DIAGONAL)
        {
            a.Shape = { 1, layer.nOutputRows };
            a.Layout[0] = 'H';
            a.Layout[1] = '\0';
        }
        else if (layer.operation == INTEL_RECURRENT)
        {
            a.Shape = { 2, layer.nOutputColumns, layer.nInputColumns + layer.nOutputColumns };
            a.Layout[0] = 'H';
            a.Layout[1] = 'W';
            a.Layout[2] = '\0';
        }
        else
        {
            a.Shape = { 2, layer.nOutputRows, layer.nInputRows };
            a.Layout[0] = 'H';
            a.Layout[1] = 'W';
            a.Layout[2] = '\0';
        }

        return a;
    }

    template<typename T>
    static Gna2Tensor getBiasTensor(const nn_layer& layer, const T& affineFunc)
    {
        Gna2Tensor a{};
        a.Layout[0] = 'H';
        a.Layout[1] = '\0';
        a.Type = DataMode(affineFunc.nBytesPerBias).Type;
        a.Data = affineFunc.pBiases;
        a.Mode = Gna2TensorModeDefault;
        a.Shape = { 1, layer.operation == INTEL_RECURRENT
                        ? layer.nOutputColumns : layer.nOutputRows };

        return a;
    }

    template<typename T>
    static T GetParameterAs(const Gna2Operation & operation, uint32_t index, T defaultValue)
    {
        if (IsParameterAvailable(operation, index))
        {
            return *static_cast<T*> (operation.Parameters[index]);
        }
        return defaultValue;
    }

    static Gna2Tensor GetOperand(const Gna2Operation & operation, uint32_t index, Gna2Tensor defaultValue);

    static Gna2BiasMode GetBiasMode(const Gna2Operation& operation);
    static Gna2BiasMode GetBiasMode(const nn_layer& layer);
    static Gna2Tensor GetBiases(const Gna2Operation& operation);
    static Gna2Tensor GetBiases(const nn_layer& layer);

    static Shape GetStride(const Gna2Operation& operation);
    static Shape GetStride(const nn_layer& layer);

    static uint32_t GetFeedbackDelay(const Gna2Operation& operation);
    static uint32_t GetFeedbackDelay(const nn_layer& layer);

    static Shape GetZeroPadding(const Gna2Operation& operation);
    static Shape GetZeroPadding(const nn_layer& layer);

    static nn_layer_pool2d GetPoolingImpl(const nn_layer& layer);
    static Shape GetPoolingStride(const nn_layer_pool2d& pooling);
    static Shape GetPoolingWindow(const nn_layer_pool2d& pooling);
    static Shape GetPoolingWindow(const Gna2Operation& operation);
    static Shape GetPoolingStride(const Gna2Operation& operation);
    static Gna2PoolingMode GetPoolingMode(const nn_layer_pool2d& pooling);

    static bool hasPooling(const Gna2Operation& operation);
    static bool hasPooling(const nn_layer& layer);

    static bool isCNN2D(const nn_layer& layer);
    static bool isCNN2D(const Gna2Operation& operation);

    static bool isAffine(const nn_layer& layer);
    static bool isAffine(const Gna2Operation& operation);

    static bool isRecurrent(const nn_layer& layer);
    static bool isRecurrent(const Gna2Operation& operation);
};

}
