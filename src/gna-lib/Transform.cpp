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

#include "ActivationHelper.h"
#include "OperationConfig.h"

#include <set>

using namespace GNA;

bool TransformFactoryConfig::HasMandatoryActivation() const
{
    return mandatoryActivation;
}

bool TransformFactoryConfig::IsActivationNotSupported() const
{
    static const std::set<nn_operation> forbiddenActivation{
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

void TransformFactoryConfig::InitActivation(const Gna2Operation & operation)
{
    mandatoryActivation = HasMandatoryActivation(operation);
    activation = GetActivation(operation);
}

inline bool TransformFactoryConfig::HasMandatoryActivation(const Gna2Operation & operation)
{
    if(OperationConfig::IsCNN1D(operation))
    {
        return Gna2PoolingModeDisabled != OperationConfig::GetPoolingMode(operation);
    }
    return operation.Type == Gna2OperationTypeRecurrent;
}

inline Gna2Tensor TransformFactoryConfig::GetActivation(const Gna2Operation & operation)
{
    if (operation.NumberOfOperands > PwlOperandIndex && operation.Type != Gna2OperationTypeGmm &&
        operation.Operands != nullptr && operation.Operands[PwlOperandIndex] != nullptr)
    {
        return *operation.Operands[PwlOperandIndex];
    }
    Gna2Tensor disabled{};
    disabled.Mode = Gna2TensorModeDisabled;
    return disabled;
}

Tensor const & BaseTransform::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case InputOperandIndex:
        if (Input)
        {
            return *Input;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case OutputOperandIndex:
        return GetOperandIfExistOrThrow(Output);
    default:
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}
