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

#include <array>
#include <vector>

using namespace GNA;

Shape::Shape(ShapeMap && map, gna_tensor_order order) :
    ShapeMap{ std::move(map) },
    LayoutOrder{ order },
    Order{ order }
{}

Shape::Shape(const gna_3d_dimensions shape) :
    Shape{ GNA_TENSOR_WHD, shape.width, shape.height, shape.depth }
{}

Shape & Shape::operator=(const Shape & right)
{
    ShapeMap::operator=(static_cast<ShapeMap>(right));
    this->Order = right.Order;
    this->LayoutOrder = right.LayoutOrder;
    return (*this);
}

uint32_t & Shape::operator[](char dimension)
{
    return ShapeMap::operator[](Layout::GetIndex(dimension));
}

uint32_t Shape::at(char dimension) const
{
    return ShapeMap::at(Layout::GetIndex(dimension));
}

Shape Shape::Reshape(gna_tensor_order order) const
{
    const Layout newLayout{ order };
    //TODO:3:P1:Check correctness after commenting out the following 2 lines
    //auto layout = LayoutOrder;
    //layout.Reshape(newLayout, size());
    ShapeMap dims;
    for (const auto & dim : newLayout)
    {
        dims[Layout::GetIndex(dim)] = this->at(dim);
    }
    return Shape(std::move(dims), order);
}

Shape Shape::Create(const ApiShape & shape, const gna_tensor_order order)
{
    // TODO:3:verify if initializer_list can always be constructed using 2 pointers
    return Shape(Create(std::vector<uint32_t>(shape.Dimensions,
        &shape.Dimensions[shape.NumberOfDimensions]), order), order);
}

ShapeMap Shape::Create(const std::vector<uint32_t> && dimensions, const gna_tensor_order order)
{
    const auto & layout = Layout(order);
    layout.ValidateNumberOfDimensions(dimensions.size());

    ShapeMap shape;
    size_type i = 0;
    for (const auto & dim : dimensions)
    {
        char index = layout.at(i++);
        shape[Layout::GetIndex(index)] = dim;
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
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
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
