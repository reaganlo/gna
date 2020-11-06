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

#include "Tensor.h"

#include "Expect.h"
#include "Macros.h"
#include "ModelError.h"
#include "Validator.h"

#include <memory>
#include <utility>
#include <vector>

using namespace GNA;

Tensor::Tensor(const ApiTensor & tensor) :
    Tensor{ Shape::Create(tensor.Shape, Layout{ tensor.Layout }),
        GetDataMode(tensor).Type, GetDataMode(tensor).Mode, tensor.Data }
{
}

DataMode Tensor::GetDataMode(const Gna2Tensor& tensor)
{
    ModelErrorHelper::ExpectInSet(tensor.Mode, { Gna2TensorModeDefault, Gna2TensorModeExternalBuffer });
    try
    {
        return DataMode{ tensor.Type, tensor.Mode };
    }
    catch(...)
    {
        throw GnaModelErrorException(Gna2ItemTypeOperandType, Gna2ErrorTypeNotInSet, tensor.Type);
    }
}

Tensor::Tensor(const ApiTensor & tensor, gna_tensor_order order, const Validator & validatorIn) :
    Tensor{ Shape::Create(tensor.Shape, order),
        GetDataMode(tensor),
        tensor.Data,
        validatorIn}
{
}

Tensor::Tensor(const Shape & dimensions, const DataType dataType, const TensorMode tensorMode, void const * buffer) :
    Component{ dimensions },
    Mode{ dataType, tensorMode },
    Size{ getEffectiveSize(Mode, Count) },
    Buffer{ buffer }
{}

Tensor::Tensor(const Shape & dimensions, const DataMode & dataMode, void const * buffer,
    const Validator & validatorIn) :
    Component{ dimensions, validatorIn, false }, // disable dimension validation as it's performed here with Mode information
    Mode{ dataMode },
    Size{ getEffectiveSize(Mode, Count) },
    Buffer{ buffer }
{
    validate();
}

Tensor::Tensor(const Tensor & tensor, const Validator & validatorIn) :
    Tensor{ tensor.Dimensions, tensor.Mode, tensor.Buffer, validatorIn }
{}

Tensor::Tensor(const ApiTensor& apiTensor, const Validator& validatorIn) :
    Tensor { Tensor{apiTensor}, validatorIn }
{}

void Tensor::UpdateBuffer(const BaseAddress & buffer)
{
    ValidateBuffer(buffer);
    Buffer = buffer;
}

void Tensor::ValidateBuffer(const void * const buffer) const
{
    auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
    validator->ValidateBuffer(buffer, Size, caps->GetAddressAlign().Value);
}

void Tensor::validate() const
{
    if (validator)
    {
        const auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
        try
        {
            Expect::InSet(Mode, caps->Modes);
        }
        catch(GnaException&)
        {
            throw GnaModelErrorException(
                Gna2ItemTypeOperandType,
                Gna2ErrorTypeNotInSet,
                Mode.Type);
        }
        // TODO:3: what about data disabled? what size and dimensions? leave dimensions as should be and mode=size=0?
        if (GNA_DATA_DISABLED != Mode)
        {
            validateDimensions();
            validator->ValidateBufferIfSet(Buffer, Size, caps->GetAddressAlign());
        }
        else
        {
            Expect::Null(Buffer);
        }
    }
}

void Tensor::validateDimensions() const
{
    // update Multiplier when varies for data modes
    auto caps = *validator->Capabilities;
    for (auto & dim : caps.Dimensions)
    {
        dim.second.Multipliers.SetEffective(Mode.Type);
    }
    Component::Validate(caps, true);
}

uint32_t Tensor::getEffectiveSize(const DataMode& mode, uint32_t count)
{
    return Gna2TensorModeConstantScalar == mode.Mode ? mode.Size : count * mode.Size;
}

std::pair<uint32_t, uint32_t> Tensor::getGroupingAndElements(
      const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    UNREFERENCED_PARAMETER(validatorIn);
    switch (operation.Type)
    {
    case Gna2OperationTypeFullyConnectedAffine:
    case Gna2OperationTypeElementWiseAffine:
    case Gna2OperationTypeThreshold:
        return {Dimensions.at('W'), Dimensions.at('H')};
    case Gna2OperationTypeRecurrent:
    case Gna2OperationTypeCopy:
    case Gna2OperationTypeGmm:
        return {Dimensions.at('H'), Dimensions.at('W')};
    case Gna2OperationTypeConvolution:
        return {1, Count}; // not applicable for 2D CNN
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}

const AlignLimits* TensorLimits::overridenAlign = nullptr;

void TensorLimits::OverrideAlign(const uint32_t newAlign)
{
    static AlignLimits overriden{ 1, Gna2StatusMemoryAlignmentInvalid };
    if (newAlign == 0)
    {
        overridenAlign = nullptr;
    }
    overriden.Value = newAlign;
    overridenAlign = &overriden;
}

const AlignLimits& TensorLimits::GetAddressAlign() const
{
    if(overridenAlign != nullptr)
    {
        return *overridenAlign;
    }
    return addressAlign;
}
