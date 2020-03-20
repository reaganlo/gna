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
#include "ModelError.h"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>

using namespace GNA;

constexpr const char LAYOUT_ANY[] = "NWHDXYZ";

const std::map<const std::string, gna_tensor_order> Layout::orderStrings
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
    { "HW", GNA_TENSOR_HW },
    { "HWD", GNA_TENSOR_HWD },
    { "NWD", GNA_TENSOR_NWD },
    { "HDW", GNA_TENSOR_HDW },
    { "NWH", GNA_TENSOR_NWH },
    { "NHW", GNA_TENSOR_NHW },
    { "WHD", GNA_TENSOR_WHD },
    { "NHWD", GNA_TENSOR_NHWD },
    { "NDHW", GNA_TENSOR_NDHW },
    { LAYOUT_ANY, GNA_TENSOR_ORDER_ANY },
};

Layout::Layout() :
    Layout{ LAYOUT_ANY }
{}

Layout::Layout(char const * layoutIn) :
    std::string{ layoutIn }
{
    auto const found = orderStrings.find(*this);
    if (orderStrings.end() == found)
    {
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
    }
}

Layout::Layout(gna_tensor_order order) :
    Layout{ GetOrderString(order) }
{}

char const * Layout::GetOrderString(gna_tensor_order order)
{
    const auto orderString = std::find_if(
          orderStrings.begin(),
          orderStrings.end(),
          [order](const auto& iter) {return iter.second == order; });
    if (orderString != orderStrings.end())
    {
        return orderString->first.c_str();
    }
    throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
}

Layout::operator gna_tensor_order() const
{
    try
    {
        return orderStrings.at(*this);
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
    }
}

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
    if (LAYOUT_ANY != *this)
    {
        ModelErrorHelper::ExpectEqual(shapeDimensions, size(), Gna2ItemTypeShapeNumberOfDimensions);
    }
    ModelErrorHelper::ExpectBelowEq(shapeDimensions, MaximumNumberOfDimension,
        Gna2ItemTypeShapeNumberOfDimensions);
}

void Layout::Reshape(Layout const & newLayout, size_type shapeDimensions)
{
    if (LAYOUT_ANY != newLayout)
    {
        ModelErrorHelper::ExpectAboveEq(shapeDimensions, size(),
            Gna2ItemTypeShapeNumberOfDimensions);
    }
    ModelErrorHelper::ExpectBelowEq(shapeDimensions, MaximumNumberOfDimension,
        Gna2ItemTypeShapeNumberOfDimensions);
    *this = newLayout;
}

int32_t Layout::GetApiIndex(char dim) const
{
    return GetApiIndex(GetIndex(dim));
}

int32_t Layout::GetApiIndex(gna_tensor_dim dim) const
{
    for (unsigned index = 0; index < size(); index++)
    {
        if (GetIndex(at(index))== dim)
        {
            return static_cast<int32_t>(index);
        }
    }
    return GNA2_DISABLED;
}
