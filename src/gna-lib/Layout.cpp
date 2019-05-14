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

#include "Layout.h"

#include "Expect.h"
#include "GnaException.h"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>

using namespace GNA;

constexpr const char LAYOUT_ANY[] = "NWHDXYZ";

Layout::Layout() :
    Layout{ LAYOUT_ANY }
{}

Layout::Layout(char const * layoutIn) :
    std::string{ layoutIn }
{
    auto const & orders = GetOrders();
    auto const found = orders.find(*this);
    if (orders.end() == found)
    {
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
    }
}

Layout::Layout(gna_tensor_order order) :
    Layout{ GetOrderString(order) }
{}

const std::map<const std::string, gna_tensor_order>& Layout::GetOrders()
{
    static const std::map<const std::string, gna_tensor_order>
        orderStrings =
    {
        { "", GNA_TENSOR_SCALAR },
        { "N", GNA_TENSOR_N },
        { "W", GNA_TENSOR_W },
        { "H", GNA_TENSOR_H },
        { "NW", GNA_TENSOR_NW },
        { "NH", GNA_TENSOR_NH },
        { "WN", GNA_TENSOR_WN },
        { "WH", GNA_TENSOR_WH },
        { "HN", GNA_TENSOR_HN },
        { "HD", GNA_TENSOR_HD },
        { "HDW", GNA_TENSOR_HDW },
        { "NWH", GNA_TENSOR_NWH },
        { "NHW", GNA_TENSOR_NHW },
        { "WHD", GNA_TENSOR_WHD },
        { "NHWD", GNA_TENSOR_NHWD },
        { "NDHW", GNA_TENSOR_NDHW },
        { LAYOUT_ANY, GNA_TENSOR_ORDER_ANY },
    };
    return orderStrings;
}

char const * Layout::GetOrderString(gna_tensor_order order)
{
    auto const & orders = GetOrders();

    const auto orderString = std::find_if(
          orders.begin(),
          orders.end(),
          [order](const auto& iter) {return iter.second == order; });
    if (orderString != orders.end())
    {
        return orderString->first.c_str();
    }
    else
    {
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
    }
}

Layout::operator gna_tensor_order() const
{
    try
    {
        return GetOrders().at(*this);
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
    }
}
//
//const std::vector<gna_tensor_dim> & Layout::GetVectorIndices(gna_tensor_order order)
//{
//    static const std::unordered_map<gna_tensor_order, const std::vector<gna_tensor_dim>, TensorDimHash>
//        vectorIndices =
//    {
//        { GNA_TENSOR_SCALAR, {} },
//        { GNA_TENSOR_N, { GNA_DIM_N } },
//        { GNA_TENSOR_W, { GNA_DIM_W } },
//        { GNA_TENSOR_H, { GNA_DIM_H } },
//        { GNA_TENSOR_NW, { GNA_DIM_N, GNA_DIM_W } },
//        { GNA_TENSOR_NH, { GNA_DIM_N, GNA_DIM_H } },
//        { GNA_TENSOR_WN, { GNA_DIM_W, GNA_DIM_N } },
//        { GNA_TENSOR_WH, { GNA_DIM_W, GNA_DIM_H } },
//        { GNA_TENSOR_HN, { GNA_DIM_H, GNA_DIM_N } },
//        { GNA_TENSOR_HD, { GNA_DIM_H, GNA_DIM_D } },
//        { GNA_TENSOR_HDW, { GNA_DIM_H, GNA_DIM_D, GNA_DIM_W } },
//        { GNA_TENSOR_NWH, { GNA_DIM_N, GNA_DIM_W, GNA_DIM_H } },
//        { GNA_TENSOR_NHW, { GNA_DIM_N, GNA_DIM_H, GNA_DIM_W } },
//        { GNA_TENSOR_WHD, { GNA_DIM_W, GNA_DIM_H, GNA_DIM_D } },
//        { GNA_TENSOR_NHWD, { GNA_DIM_N, GNA_DIM_H, GNA_DIM_W, GNA_DIM_D } },
//        { GNA_TENSOR_NDHW, { GNA_DIM_N, GNA_DIM_D, GNA_DIM_H, GNA_DIM_W } },
//        {
//            GNA_TENSOR_ORDER_ANY,
//            {
//                GNA_DIM_N,
//                GNA_DIM_W,
//                GNA_DIM_H,
//                GNA_DIM_D,
//                GNA_DIM_X,
//                gna_tensor_dim(GNA_DIM_X + 1),
//                gna_tensor_dim(GNA_DIM_X + 2),
//                gna_tensor_dim(GNA_DIM_X + 3)
//            }
//        },
//    };
//
//    try
//    {
//        return vectorIndices.at(order);
//    }
//    catch (const std::out_of_range&)
//    {
//        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
//    }
//}

//char Layout::GetIndex(gna_tensor_dim dim)
//{
//    static const std::unordered_map<gna_tensor_dim, char> indices =
//    {
//        { GNA_DIM_N, 'N' },
//        { GNA_DIM_W, 'W' },
//        { GNA_DIM_H, 'H' },
//        { GNA_DIM_D, 'D' },
//        { GNA_DIM_X, 'X' },
//        { GNA_DIM_Y, 'Y' },
//        { GNA_DIM_Z, 'Z' },
//    };
//
//    try
//    {
//        return indices.at(dim);
//    }
//    catch (const std::out_of_range&)
//    {
//        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
//    }
//}

gna_tensor_dim Layout::GetIndex(char dim)
{
    static const std::unordered_map<char, gna_tensor_dim> indices =
    {
        { 'N', GNA_DIM_N },
        { 'W', GNA_DIM_W },
        { 'H', GNA_DIM_H },
        { 'D', GNA_DIM_D },
        { 'X', GNA_DIM_X },
        { 'Y', GNA_DIM_Y },
        { 'Z', GNA_DIM_Z },
    };

    try
    {
        return indices.at(dim);
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
    }
}

void Layout::ValidateNumberOfDimensions(size_type shapeDimensions) const
{
    auto condition = true;
    if (LAYOUT_ANY != *this)
    {
        condition = shapeDimensions == size();
    }
    Expect::True(shapeDimensions <= MaximumNumberOfDimension && condition,
        Gna2StatusModelConfigurationInvalid); // TODO:3: add error code
}

void Layout::Reshape(Layout const & newLayout, size_type shapeDimensions)
{
    auto condition = true;
    if (LAYOUT_ANY != newLayout)
    {
        condition = shapeDimensions >= size();
    }
    Expect::True(shapeDimensions <= MaximumNumberOfDimension && condition,
        Gna2StatusModelConfigurationInvalid); // TODO:3: add error code
    *this = newLayout;
}
