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

using namespace GNA;

const std::map<const gna_tensor_order, const std::vector<gna_tensor_dim>> Shape::VectorIndices
{
    {GNA_TENSOR_W, {GNA_DIM_W}},
    {GNA_TENSOR_H, {GNA_DIM_H}},
    {GNA_TENSOR_NW, {GNA_DIM_N, GNA_DIM_W}},
    {GNA_TENSOR_NH, {GNA_DIM_N, GNA_DIM_H}},
    {GNA_TENSOR_WN, {GNA_DIM_W, GNA_DIM_N}},
    {GNA_TENSOR_WH, {GNA_DIM_W, GNA_DIM_H}},
    {GNA_TENSOR_HN, {GNA_DIM_H, GNA_DIM_N}},
    {GNA_TENSOR_HD, {GNA_DIM_H, GNA_DIM_D}},
    {GNA_TENSOR_HDW, {GNA_DIM_H, GNA_DIM_D, GNA_DIM_W}},
    {GNA_TENSOR_NWH, {GNA_DIM_N, GNA_DIM_W, GNA_DIM_H}},
    {GNA_TENSOR_WHD, {GNA_DIM_W, GNA_DIM_H, GNA_DIM_D}},
    {GNA_TENSOR_NHWD, {GNA_DIM_N, GNA_DIM_H, GNA_DIM_W, GNA_DIM_D}},
    {GNA_TENSOR_NDHW, {GNA_DIM_N, GNA_DIM_D, GNA_DIM_H, GNA_DIM_W}},
    {GNA_TENSOR_ORDER_ANY, {GNA_DIM_N, GNA_DIM_W, GNA_DIM_H, GNA_DIM_D,
        GNA_DIM_X, gna_tensor_dim(GNA_DIM_X + 1),
        gna_tensor_dim(GNA_DIM_X + 2), gna_tensor_dim(GNA_DIM_X + 3)}},
};

Shape::Shape() :
    __ShapeMap(),
    order{GNA_TENSOR_SCALAR}
{}

Shape::Shape(const uint32_t x, const uint32_t y, gna_tensor_order mapOrder) :
    __ShapeMap({ { VectorIndices.at(mapOrder).at(0), x }, { VectorIndices.at(mapOrder).at(1), y } }),
    order{mapOrder}
{
    Expect::Equal(VectorIndices.at(mapOrder).size(), (size_t)2, XNN_ERR_NETWORK_INPUTS); // TODO:3: add error code
}

Shape::Shape(const uint32_t x, const uint32_t y, const uint32_t z, gna_tensor_order mapOrder) :
    __ShapeMap({ { VectorIndices.at(mapOrder).at(0), x },
          { VectorIndices.at(mapOrder).at(1), y },
          { VectorIndices.at(mapOrder).at(2), z } }),
    order{mapOrder}
{
    Expect::Equal(VectorIndices.at(mapOrder).size(), (size_t)3, XNN_ERR_NETWORK_INPUTS); // TODO:3: add error code
}

Shape::Shape(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w, gna_tensor_order mapOrder) :
    __ShapeMap({ { VectorIndices.at(mapOrder).at(0), x },
          { VectorIndices.at(mapOrder).at(1), y },
          { VectorIndices.at(mapOrder).at(2), z },
          { VectorIndices.at(mapOrder).at(3), w }}),
    order{mapOrder}
{
    // TODO:3: will never hit as exception is generated above
    Expect::Equal(VectorIndices.at(mapOrder).size(), (size_t)4, XNN_ERR_NETWORK_INPUTS); // TODO:3: add error code
}

Shape::Shape(const uint32_t N, const uint32_t W, const uint32_t H) :
    Shape{N, W, H, GNA_TENSOR_NWH}
{};

Shape::Shape(const Shape map, gna_tensor_order newOrder) :
    __ShapeMap{SubsetDimensionMap(map, newOrder)},
    order{newOrder}
{};

Shape::Shape(const gna_3d_dimensions shape) :
    Shape{shape.width, shape.height, shape.depth, GNA_TENSOR_WHD}
{}

Shape::Shape(const ApiShape & apiShape, const gna_tensor_order layout)  :
    __ShapeMap{ Create(apiShape, layout) },
    order{ layout }
{};

Shape& Shape::operator=(const Shape& right)
{
    __ShapeMap::operator=(static_cast<__ShapeMap>(right));
    this->order = right.order;
    return (*this);
}

const Shape Shape::SubsetDimensionMap(
    const __ShapeMap& shape, gna_tensor_order newOrder)
{
    Expect::True(shape.size() >= VectorIndices.at(newOrder).size(), XNN_ERR_NETWORK_INPUTS); // TODO:3: add error code
    Shape dims;
    for (const auto& dim : VectorIndices.at(newOrder))
    {
        dims[dim] = shape.at(dim);
    }
    return dims;
}

__ShapeMap Shape::Create(const ApiShape & apiShape, const gna_tensor_order order)
{
    const auto dimensionCount = static_cast<size_type>(apiShape.NumberOfDimensions);
    Expect::InRange<size_type>(dimensionCount,
        0, ShapeMaximumNumberOfDimension,
        CAST1_STATUS Gna2StatusModelConfigurationInvalid);
    
    const auto& layout = VectorIndices.at(order);
    Expect::InRange(dimensionCount, layout.size(),
        CAST1_STATUS Gna2StatusModelConfigurationInvalid);
    
    __ShapeMap shape;
    size_type i = 0;
    for (; i < dimensionCount; i++)
    {
        shape[layout.at(i)] = apiShape.Dimensions[i];
    }
    for (; i < layout.size(); i++)
    {
        shape[layout.at(i)] = 0;
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

uint32_t Shape::GetNumberOfElements() const
{
    uint32_t count = 1;
    uint32_t sum = 0;
    for (const auto& dim : *this)
    {
        sum += dim.second;
        if (0 != dim.second)
        {
            count *= dim.second;
        }
    }
    if (0 == sum)
    {
        return 0;
    }
    else
    {
        return count;
    }
}

void Shape::Validate(const ComponentLimits* validator) const
{
    Expect::Equal(order, validator->Order.Value, validator->Order.Error);
    Expect::ShapeIsValid(*this, validator->Dimensions);
}
