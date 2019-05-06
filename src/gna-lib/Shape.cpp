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

#include "Shape.h"

#include "Expect.h"

#include <unordered_map>
#include <vector>
#include <array>

using namespace GNA;

const std::unordered_map<gna_tensor_order, const std::vector<gna_tensor_dim>, TensorDimHash> Shape::VectorIndices
{
    { GNA_TENSOR_SCALAR, {} },
    { GNA_TENSOR_W, { GNA_DIM_W } },
    { GNA_TENSOR_H, { GNA_DIM_H } },
    { GNA_TENSOR_NW, { GNA_DIM_N, GNA_DIM_W } },
    { GNA_TENSOR_NH, { GNA_DIM_N, GNA_DIM_H } },
    { GNA_TENSOR_WN, { GNA_DIM_W, GNA_DIM_N } },
    { GNA_TENSOR_WH, { GNA_DIM_W, GNA_DIM_H } },
    { GNA_TENSOR_HN, { GNA_DIM_H, GNA_DIM_N } },
    { GNA_TENSOR_HD, { GNA_DIM_H, GNA_DIM_D } },
    { GNA_TENSOR_HDW, { GNA_DIM_H, GNA_DIM_D, GNA_DIM_W } },
    { GNA_TENSOR_NWH, { GNA_DIM_N, GNA_DIM_W, GNA_DIM_H } },
    { GNA_TENSOR_WHD, { GNA_DIM_W, GNA_DIM_H, GNA_DIM_D } },
    { GNA_TENSOR_NHWD, { GNA_DIM_N, GNA_DIM_H, GNA_DIM_W, GNA_DIM_D } },
    { GNA_TENSOR_NDHW, { GNA_DIM_N, GNA_DIM_D, GNA_DIM_H, GNA_DIM_W } },
    {
        GNA_TENSOR_ORDER_ANY,
        {
            GNA_DIM_N,
            GNA_DIM_W,
            GNA_DIM_H,
            GNA_DIM_D,
            GNA_DIM_X,
            gna_tensor_dim(GNA_DIM_X + 1),
            gna_tensor_dim(GNA_DIM_X + 2),
            gna_tensor_dim(GNA_DIM_X + 3)
        }
    },
};

Shape::Shape(ShapeMap && map, gna_tensor_order order) :
    ShapeMap{ std::move(map) },
    Order{ order }
{}

void Shape::ValidateNumberOfDimensions(gna_tensor_order order,
    size_type orderDimensions, size_type shapeDimensions)
{
    auto condition = false;
    if (GNA_TENSOR_ORDER_ANY == order)
    {
        condition = shapeDimensions <= orderDimensions;
    }
    else
    {
        condition = shapeDimensions == orderDimensions;
    }
    Expect::True(condition, CAST1_STATUS Gna2StatusModelConfigurationInvalid); // TODO:3: add error code
}

Shape::Shape(const gna_3d_dimensions shape) :
    Shape{ GNA_TENSOR_WHD, shape.width, shape.height, shape.depth }
{}

Shape & Shape::operator=(const Shape & right)
{
    ShapeMap::operator=(static_cast<ShapeMap>(right));
    this->Order = right.Order;
    return (*this);
}

Shape Shape::Reshape(gna_tensor_order order) const
{
    Expect::True(this->size() >= VectorIndices.at(order).size(), XNN_ERR_NETWORK_INPUTS); // TODO:3: add error code
    Shape dims;
    dims.Order = order;
    for (const auto & dim : VectorIndices.at(order))
    {
        dims[dim] = this->at(dim);
    }
    return dims;
}

Shape Shape::Create(const ApiShape & shape, const gna_tensor_order order)
{
    // TODO:3:verify if initializer_list can always be constructed using 2 pointers
    return Shape(Create(std::vector<uint32_t>(shape.Dimensions,
                &shape.Dimensions[shape.NumberOfDimensions]), order), order);
}

ShapeMap Shape::Create(const std::vector<uint32_t> && dimensions, const gna_tensor_order order)
{
    const auto & layout = VectorIndices.at(order);
    ValidateNumberOfDimensions(order, layout.size(), dimensions.size());

    ShapeMap shape;
    size_type i = 0;
    for (const auto & dim : dimensions)
    {
        shape[layout.at(i++)] = dim;
    }
    return shape;
}

Shape::operator gna_3d_dimensions const() const
{
    if (this->count(GNA_DIM_W) > 0 && this->count(GNA_DIM_H) > 0)
    {
        if (this->count(GNA_DIM_D))
        {
            return gna_3d_dimensions{at(GNA_DIM_W), at(GNA_DIM_H), at(GNA_DIM_D)};
        }
        else
        {
            return gna_3d_dimensions{at(GNA_DIM_W), at(GNA_DIM_H), 0};
        }
    }
    else
    {
        throw GnaException(XNN_ERR_LYR_INVALID_TENSOR_ORDER);
    }
}

Shape::operator ApiShape() const
{
    ApiShape shape = {};
    shape.NumberOfDimensions = static_cast<uint32_t>(size());
    uint32_t i = 0;
    for (const auto & dim : *this)
    {
        shape.Dimensions[i++] = dim.second;
    }
    return shape;
}

uint32_t Shape::GetNumberOfElements() const
{
    uint32_t count = 1;
    uint32_t sum = 0;
    for (const auto & dim : *this)
    {
        sum += dim.second;
        if (0 != dim.second)
        {
            count *= dim.second;
        }
    }
    if (0 == sum)
    {
        return 0; // TODO:3:API: should scalar return 0?
    }
    else
    {
        return count;
    }
}

void Shape::Validate(const ComponentLimits * validator) const
{
    Expect::Equal(Order, validator->Order.Value, validator->Order.Error);
    Expect::ShapeIsValid(*this, validator->Dimensions);
}
