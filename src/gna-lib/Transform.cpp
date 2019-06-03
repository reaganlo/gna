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

#include "Transform.h"

#include "OperationConfig.h"

#include <set>

using namespace GNA;

bool TransformFactoryConfig::HasMandatoryActivation() const
{
    return mandatoryActivation;
}

bool TransformFactoryConfig::IsActivationNotSupported() const
{
    const static std::set<nn_operation> forbiddenActivation{
        GNA_LAYER_CNN_2D_POOLING,
        INTEL_INTERLEAVE,
        INTEL_COPY,
        INTEL_DEINTERLEAVE };
    return 0 < forbiddenActivation.count(validator.Operation);
}

Gna2Tensor TransformFactoryConfig::GetActivation() const
{
    return activation;
}

Gna2Tensor TransformFactoryConfig::GetActivation(const void * layerDetails, nn_operation operationType)
{
    auto pwl = GetActivationImpl(layerDetails, operationType);
    Gna2Tensor a{};
    a.Type = Gna2DataTypePwlSegment;
    a.Shape = { 1, pwl.nSegments };
    a.Data = pwl.pSegments;
    return a;
}

//TODO:3:P1:Move to operation/model wrapper
void TransformFactoryConfig::InitActivation(const nn_layer & layer)
{
    mandatoryActivation = HasMandatoryActivation(layer.pLayerStruct);
    activation = GetActivation(layer.pLayerStruct, validator.Operation);
}

void TransformFactoryConfig::InitActivation(const Gna2Operation & operation)
{
    mandatoryActivation = HasMandatoryActivation(operation);
    activation = GetActivation(operation);
}

inline bool TransformFactoryConfig::HasMandatoryActivation(const void * layerDetails) const
{
    if (validator.Operation == INTEL_CONVOLUTIONAL)
    {
        auto cnn = static_cast<nn_layer_conv const*>(layerDetails);
        if (INTEL_NO_POOLING != cnn->poolType)
            return true;
    }
    return validator.Operation == INTEL_RECURRENT;
}

inline bool TransformFactoryConfig::HasMandatoryActivation(const Gna2Operation & operation)
{
    if(OperationConfig::IsCNN1D(operation))
    {
        return Gna2PoolingModeDisabled != OperationConfig::GetPoolingMode(operation);
    }
    return operation.Type == Gna2OperationTypeRecurrent;
}

inline nn_func_pwl TransformFactoryConfig::GetActivationImpl(const void * layerDetails, nn_operation operationType)
{
    switch (operationType)
    {
    case INTEL_AFFINE: /* FALLTHRU */
    case INTEL_AFFINE_DIAGONAL:
        return static_cast<nn_layer_affine const*>(layerDetails)->pwl;
    case INTEL_AFFINE_MULTIBIAS:
        return static_cast<nn_layer_affine_multi const*>(layerDetails)->pwl;
    case INTEL_CONVOLUTIONAL:
        return static_cast<nn_layer_conv const*>(layerDetails)->pwl;
    case INTEL_CONVOLUTIONAL_2D:
        return static_cast<nn_layer_cnn2d const*>(layerDetails)->activation;
    case INTEL_RECURRENT:
        return static_cast<nn_layer_recurrent const*>(layerDetails)->pwl;
    default:
        throw GnaException{ Gna2StatusXnnErrorLyrOperation };
    }
}

inline Gna2Tensor TransformFactoryConfig::GetActivation(const Gna2Operation & operation)
{
    if (operation.NumberOfOperands >= 5 && operation.Type != Gna2OperationTypeGmm &&
        operation.Operands != nullptr && operation.Operands[4] != nullptr)
    {
        return *operation.Operands[4];
    }
    Gna2Tensor a{};
    a.Mode = Gna2TensorModeDisabled;
    return a;
}
