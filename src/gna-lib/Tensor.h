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

#pragma once

#include "Component.h"
#include "DataMode.h"

namespace GNA
{

struct Tensor : public Component
{
    Tensor(const Shape& dimensions, const DataMode& dataMode, void const * buffer,
        const Validator& validator);
    virtual ~Tensor() = default;

    void UpdateBuffer(const BaseAddress& outputBuffer);

    void ValidateBuffer(const void* const buffer) const;

    virtual operator const BaseAddress& () const
    {
        return Buffer;
    }

    virtual operator void* () const
    {
        return Buffer;
    }

    const DataMode Mode;

    // Total size in bytes of tensor data buffer
    const uint32_t Size;

    BaseAddress Buffer;

protected:
    void validate() const;
private:
    void validateDimensions() const;
};

struct TensorLimits : public ComponentLimits
{
    TensorLimits(const ComponentLimits limits, const DataModeLimits& modes) :
        ComponentLimits{ limits },
        Modes{ modes },
        Align{ GNA_MEM_ALIGN, GNA_BADMEMALIGN }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        Align{ GNA_MEM_ALIGN, GNA_BADMEMALIGN }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes,
        const AlignLimits& align) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        Align{ align }
    {
    }

    const DataModeLimits Modes;
    const AlignLimits Align;
};

};




