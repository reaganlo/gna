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

#include "Component.h"
#include "Expect.h"
#include "Layout.h"
#include "ParameterLimits.h"

using namespace GNA;

Component::Component(const Shape & dimensions) :
    Dimensions{ dimensions },
    Count{ Dimensions.GetNumberOfElements() },
    validator{ nullptr }
{
}

Component::Component(const Component & component, const Validator & validatorIn, bool validateDimensions) :
    Dimensions{ component.Dimensions },
    Count{ component.Count }
{
    validator = std::make_unique<const Validator>(validatorIn);
    Expect::NotNull(validator.get());
    Validate(*validator->Capabilities, validateDimensions);
}

Component::Component(const Shape& dimensions, const Validator& validatorIn, bool validateDimensions) :
    Component{ Component{ dimensions.Reshape(validatorIn.Order) }, validatorIn, validateDimensions }
{}

nn_operation Component::GetEffectiveOperationType() const
{
    if (validator)
    {
        return validator->Operation;
    }
    return LAYER_OPERATION_TYPE_COUT;
}

void Component::Validate(const ComponentLimits& limits, bool validateDimensions) const
{
    if (validateDimensions)
    {
        Expect::Equal(Dimensions.LayoutOrder.operator _tensor_order(),
            limits.Order.Value, limits.Order.Error);
        Expect::ShapeIsValid(Dimensions, limits.Dimensions);
    }
}
