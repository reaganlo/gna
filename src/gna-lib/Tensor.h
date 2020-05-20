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
#include "Component.h"
#include "DataMode.h"
#include "ParameterLimits.h"
#include "Shape.h"

#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <cstdint>

namespace GNA
{
class Validator;

struct Tensor : public Component
{
    Tensor(const ApiTensor& tensor);

    Tensor(const ApiTensor& tensor, gna_tensor_order order, const Validator& validator);

    Tensor(const ApiTensor& apiTensor, const Validator& validatorIn);

    Tensor(const Shape& dimensions, const DataType dataType,
        const TensorMode tensorMode, void const * buffer);

    Tensor(const Shape& dimensions, const DataMode& dataMode,
        void const * buffer, const Validator& validatorIn);

    virtual ~Tensor() = default;

    void UpdateBuffer(const BaseAddress& buffer);

    void ValidateBuffer(const void* const buffer) const;

    virtual operator const BaseAddress() const
    {
        return Buffer;
    }

    virtual operator void* () const
    {
        return Buffer;
    }

    explicit operator ApiTensor() const
    {
        ApiTensor tensor{};
        tensor.Shape = Dimensions;
        tensor.Mode = Mode.Mode;
        tensor.Type = Mode.Type;
        tensor.Data = Buffer;
        if (Layout() != Dimensions.LayoutOrder)
        {
            snprintf(tensor.Layout, sizeof(tensor.Layout), "%s", Dimensions.LayoutOrder.c_str());
        }
        return tensor;
    }

    static DataMode GetDataMode(const Gna2Tensor& tensor);

    // TODO:3:API: remove and use Type and Mode directly
    const DataMode Mode;

    // Total size in bytes of tensor data buffer
    const uint32_t Size;

    BaseAddress Buffer;

    static Shape GetDimensions(const ApiTensor& operand, gna_tensor_order order)
    {
        return Shape::Create(operand.Shape, order);
    }

protected:
    Tensor(const Tensor& tensor, const Validator& validatorIn);

    void validate() const;

    uint32_t getGrouping(const Gna2Operation& operation, const LayerValidator& validatorIn) const
    {
        return getGroupingAndElements(operation, validatorIn).first;
    }

    uint32_t getGrouping(const nn_layer& layer) const
    {
        return getGroupingAndElements(layer).first;
    }

    uint32_t getElementCount(const Gna2Operation& operation, const LayerValidator& validatorIn) const
    {
        return getGroupingAndElements(operation, validatorIn).second;
    }

    uint32_t getElementCount(const nn_layer& layer) const
    {
        return getGroupingAndElements(layer).second;
    }

    // Returns pair<grouping, elementCount>
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const;
    // Returns pair<grouping, elementCount>
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(const nn_layer& layer) const;

private:
    static uint32_t getEffectiveSize(const DataMode& mode, uint32_t count);

    void validateDimensions() const;
};

struct TensorLimits : public ComponentLimits
{
    TensorLimits(const ComponentLimits limits, const DataModeLimits& modes) :
        ComponentLimits{ limits },
        Modes{ modes },
        addressAlign{ GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        addressAlign{ GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes,
        const AlignLimits& align) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        addressAlign{ align }
    {
    }

    const AlignLimits& GetAddressAlign() const;
    static void OverrideAlign(const uint32_t newAlign);
    const DataModeLimits Modes;
private:
    const AlignLimits addressAlign;
    static const AlignLimits* overridenAlign;
};

}




