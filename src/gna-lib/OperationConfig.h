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

#include "DataMode.h"
#include "Shape.h"
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

    Gna2Tensor FiltersTensor;
    Shape ConvolutionStride;
    Shape ZeroPadding;
    Shape PoolingWindow;
    Shape PoolingStride;
    Gna2PoolingMode Mode;
    Gna2Tensor BiasesTensor;
    Gna2BiasMode BiasMode;

protected:
    template<class T>
    void InitOperationConfig(const T& operation)
    {
        if (isCNN2D(operation))
        {
            FiltersTensor = GetFilters(operation);
            ConvolutionStride = GetStride(operation);
            ZeroPadding = GetZeroPadding(operation);
            BiasesTensor = GetBiases(operation);
            BiasMode = GetBiasMode(operation);
        }
        if(hasPooling(operation))
        {
            InitPooling(operation);
        }
        else
        {
            Mode = Gna2PoolingModeDisabled;
        }
    }
    void InitPooling(const Gna2Operation& operation);
    void InitPooling(const nn_layer& layer);
private:

    static Shape TryGetParamShape(const Gna2Operation & operation, OperationInfoKey parameter);
    static Shape TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex);

    static const nn_layer_cnn2d* CastToCnn2DDetails(const nn_layer& layer);

    static Gna2Tensor GetFilters(const nn_layer& layer);
    static Gna2Tensor GetFilters(const Gna2Operation& operation);

    static bool IsOperandAvailable(const Gna2Operation & operation, uint32_t index);
    static bool IsParameterAvailable(const Gna2Operation & operation, uint32_t index);

    template<class T>
    static T GetParameterAs(const Gna2Operation & operation, OperationInfoKey parameter, T defaultValue)
    {
        auto const parameterIndex = ModelWrapper::GetOperationInfo(operation.Type, parameter);
        return GetParameterAs(operation, parameterIndex, defaultValue);
    }

    template<class T>
    static T GetParameterAs(const Gna2Operation & operation, uint32_t index, T defaultValue)
    {
        if (IsParameterAvailable(operation, index))
        {
            return *static_cast<T*> (operation.Parameters[index]);
        }
        return defaultValue;
    }

    static Gna2Tensor GetOperand(const Gna2Operation & operation, GnaComponentType operand, Gna2Tensor defaultValue);
    static Gna2Tensor GetOperand(const Gna2Operation & operation, uint32_t index, Gna2Tensor defaultValue);

    static Gna2BiasMode GetBiasMode(const Gna2Operation& operation);
    static Gna2BiasMode GetBiasMode(const nn_layer& layer);
    static Gna2Tensor GetBiases(const Gna2Operation& operation);
    static Gna2Tensor GetBiases(const nn_layer& layer);

    static Shape GetStride(const Gna2Operation& operation);
    static Shape GetStride(const nn_layer& layer);

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
};

}
