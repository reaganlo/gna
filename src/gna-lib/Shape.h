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

#include "gna2-model-impl.h"

#include "gna-api.h"
#include "gna2-common-impl.h"
#include "gna2-model-api.h"

#include <map>
#include <unordered_map>
#include <vector>

namespace GNA
{
struct ComponentLimits;

using ShapeMap = std::map<gna_tensor_dim, uint32_t>;

struct Shape : public ShapeMap
{
    static constexpr size_type ShapeMaximumNumberOfDimension = GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS;

    static Shape Create(const ApiShape & shape, gna_tensor_order order = GNA_TENSOR_ORDER_ANY);

    // Clang issue workaround @see: https://stackoverflow.com/questions/34494765/interaction-between-default-arguments-and-parameter-pack-gcc-and-clang-disagree
    Shape() :
        ShapeMap(),
        Order{ GNA_TENSOR_SCALAR }
    { }

    template<typename ... T>
    Shape(gna_tensor_order order, T ... dimensions) :
        Shape{ Create(std::vector<uint32_t>({ static_cast<uint32_t>(dimensions)... }), order), order }
    { }

    Shape(gna_3d_dimensions shape);

    Shape & operator=(const Shape & right);

    operator gna_3d_dimensions const() const;

    operator ApiShape() const;

    Shape Reshape(gna_tensor_order newOrder) const;

    uint32_t GetNumberOfElements() const;

    void Validate(const ComponentLimits * validator) const;

protected:
    static const std::vector<gna_tensor_dim>& GetVectorIndices(gna_tensor_order order);

    static ShapeMap Create(const std::vector<uint32_t> && dimensions,
        gna_tensor_order order = GNA_TENSOR_ORDER_ANY);

    static void ValidateNumberOfDimensions(gna_tensor_order order, size_type orderDimensions,
        size_type shapeDimensions);

    Shape(ShapeMap && map, gna_tensor_order order);

    gna_tensor_order Order;
};
}
