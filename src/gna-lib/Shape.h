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
#include "gna2-model-api.h"

#include <map>
#include <vector>

namespace GNA
{

struct ComponentLimits;

using __ShapeMap = std::map<const gna_tensor_dim, uint32_t>;

struct Shape : public __ShapeMap
{
    static constexpr size_type ShapeMaximumNumberOfDimension = GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS;

    Shape();

    Shape(const uint32_t x, const uint32_t y, gna_tensor_order mapOrder);

    Shape(const uint32_t x, const uint32_t y, const uint32_t z,
        gna_tensor_order mapOrder);

    Shape(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w,
        gna_tensor_order mapOrder);

    // Creates Shape with GNA_TENSOR_NWH order
    Shape(const uint32_t N, const uint32_t W, const uint32_t H);

    Shape(const Shape map, gna_tensor_order newOrder);

    Shape(const gna_3d_dimensions shape);

    explicit Shape(const ApiShape& shape, const gna_tensor_order layout = GNA_TENSOR_ORDER_ANY);

    Shape& operator=(const Shape& right);

    operator gna_3d_dimensions const() const;

    uint32_t GetNumberOfElements() const;

    void Validate(const ComponentLimits* validator) const;

protected:
    static const std::map<const gna_tensor_order, const std::vector<gna_tensor_dim>> VectorIndices;

    static const Shape SubsetDimensionMap(const __ShapeMap&, gna_tensor_order newOrder);

    static __ShapeMap Create(const ApiShape& shape, gna_tensor_order order);

    gna_tensor_order order;
};

}
