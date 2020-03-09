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

#include "Shape.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"

#include <memory>

namespace GNA
{
struct ComponentLimits;

struct Component
{
    Component(const Shape& dimensions);

    Component(const Component& component, const Validator& validator, bool validateDimensions = true);

    Component(const Shape& dimensions, const Validator& validator, bool validateDimensions = true);

    virtual ~Component() = default;

    nn_operation GetEffectiveOperationType() const;

    /**
     * Gets Dimensions value at dimension key
     */
    inline uint32_t at(const gna_tensor_dim dimension) const
    {
        return Dimensions.at(dimension);
    }

    virtual ModelValue AsModelValue(char dimension) const;

    Shape Dimensions;

    // Total number of elements
    uint32_t Count;

    void Validate(const FullCapabilitiesMap & caps, nn_operation operation) const;

protected:
    void Validate(const ComponentLimits& limits, bool validateDimensions = true) const;

    std::unique_ptr<const Validator> validator;
};

}
