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
    OperationConfig(const Gna2Operation& apiOperation);

    static Gna2PoolingMode GetPoolingMode(const Gna2Operation& operation);
    static bool IsCNN1D(const Gna2Operation & operation);

    static bool IsMultibias(const Gna2Operation& operation);

    static Gna2OperationType GetOperationType(const Gna2Operation& operation);

    kernel_op GetKernelOperation() const;
    TransformOperation GetTransformOperation() const;

    Gna2Tensor GetEnabledOperand(uint32_t index) const;

    template<typename T>
    T GetParameterAs(uint32_t index) const
    {
        Expect::NotNull(Operation);
        return ModelWrapper::GetParameter<T>(*Operation, index);
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
    Gna2Tensor WeightScalesTensor = ModelWrapper::GetDisabledOperand();
    uint32_t BiasVectorIndex;

    Gna2Operation const * const Operation;

protected:
    void InitOperationConfig(const Gna2Operation& operation)
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

    void InitMultibias(const Gna2Operation& operation);

private:
    static Shape TryGetParamShape(const Gna2Operation & operation, OperationInfoKey parameter);
    static Shape TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex);

    static const nn_layer_cnn2d* CastToCnn2DDetails(const nn_layer& layer);

    static Gna2Tensor GetWeights(const Gna2Operation& operation);
    static Gna2Tensor GetFilters(const Gna2Operation& operation);

    static Gna2BiasMode GetBiasMode(const Gna2Operation& operation);
    static Gna2Tensor GetBiases(const Gna2Operation& operation);

    static Shape GetStride(const Gna2Operation& operation);

    static uint32_t GetFeedbackDelay(const Gna2Operation& operation);

    static Shape GetZeroPadding(const Gna2Operation& operation);

    static Shape GetShapeParameterOfMaximal2Dimensions(const Gna2Operation& operation, uint32_t parameterIndex);

    static bool hasPooling(const Gna2Operation& operation);

    static bool isCNN2D(const Gna2Operation& operation);

    static bool isAffine(const Gna2Operation& operation);

    static bool isRecurrent(const Gna2Operation& operation);
};

}
