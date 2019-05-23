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
#include "Validator.h"

#include <memory>
#include <utility>
#include <vector>

using namespace GNA;

Tensor::Tensor(const ApiTensor & tensor) :
    Tensor{ Shape::Create(tensor.Shape, Layout{ tensor.Layout }),
        tensor.Type, tensor.Mode, tensor.Data }
{
}

Tensor::Tensor(const ApiTensor & tensor, gna_tensor_order order, const Validator & validatorIn) :
    Tensor{ Shape::Create(tensor.Shape, order),
        tensor.Type,
        tensor.Data,
        validatorIn}
{
}

Tensor::Tensor(const Shape & dimensions, const DataType dataType, const TensorMode tensorMode, void const * buffer) :
    Component{ dimensions },
    Mode{ dataType, tensorMode },
    Size{ Count * Mode.Size },
    Buffer{ buffer }
{}

Tensor::Tensor(const Shape & dimensions, const DataMode & dataMode, void const * buffer,
    const Validator & validatorIn) :
    Component{ dimensions, validatorIn, false }, // disable dimension validation as it's performed here with Mode information
    Mode{ dataMode },
    Size{ Count * Mode.Size },
    Buffer{ buffer }
{
    validate();
}

void Tensor::UpdateBuffer(const BaseAddress & buffer)
{
    ValidateBuffer(buffer);
    Buffer = buffer;
}

void Tensor::ValidateBuffer(const void * const buffer) const
{
    auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
    validator->ValidateBuffer(buffer, Size, caps->Align);
}

void Tensor::validate() const
{
    if (validator)
    {
        const auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
        Expect::InSet(Mode, caps->Modes);
        // TODO:3: what about data disabled? what size and dimensions? leave dimensions as should be and mode=size=0?
        if (GNA_DATA_DISABLED != Mode)
        {
            validateDimensions();
            validator->ValidateBufferIfSet(Buffer, Size, caps->Align);
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
        uint32_t tmp = Mode.Size;
        int log2size = 0;
        while ((tmp >>= 1) > 0)
        {
            ++log2size;
        }
        auto sizeIndex = static_cast<uint32_t>(log2size);
        if (dim.second.Multipliers.size() >= sizeIndex + 1)
        {
            dim.second.Multipliers[0] = dim.second.Multipliers.at(sizeIndex);
        }
    }
    Component::Validate(caps, true);
}
