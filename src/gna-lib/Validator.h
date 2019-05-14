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

#include "Address.h"
#include "Capabilities.h"
#include "HardwareCapabilities.h"

#include <memory>
#include <functional>

namespace GNA
{

class FullCapabilitiesMap;
struct ComponentLimits;

// Functor for validating if buffer is within memory boundaries
using ValidBoundariesFunctor = std::function<void(const void *, size_t)>;

class BaseValidator
{
public:
    BaseValidator(
        const HardwareCapabilities hwCapabilities,
        const ValidBoundariesFunctor bufferValidator);
    virtual ~BaseValidator() = default;

    void ValidateBuffer(const void* const buffer, size_t size,
        const AlignLimits& alignLimits = {GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid}) const;

    inline void ValidateBufferIfSet(const void* const buffer, size_t size,
        const AlignLimits& alignLimits = {GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid}) const
    {
        if (buffer)
            ValidateBuffer(buffer, size, alignLimits);
    }

    const HardwareCapabilities HwCapabilities;

protected:
    const ValidBoundariesFunctor bufferValidator;
};

class LayerValidator : public BaseValidator
{
public:
    LayerValidator(const BaseValidator& validator, nn_operation operation);
    virtual ~LayerValidator() = default;

    const nn_operation Operation;
};

class Validator : public LayerValidator
{
public:
    Validator(const LayerValidator& validator, const FullCapabilitiesMap& capabilities);
    virtual ~Validator() = default;

    const ComponentLimits * const Capabilities;
    const gna_tensor_order Order;
};

};
