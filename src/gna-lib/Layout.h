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

#include "gna-api.h"
#include "gna2-model-api.h"
#include "ParameterLimits.h"

#include <map>
#include <string>

namespace GNA
{

class Layout : public std::string
{
public:
    static constexpr size_type MaximumNumberOfDimension = GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS;
    //static const std::vector<gna_tensor_dim> & GetVectorIndices(gna_tensor_order order);
    //static char GetIndex(gna_tensor_dim dim);
    static gna_tensor_dim GetIndex(char dim);

    // Creates default layout as LAYOUT_ANY
    Layout();
    Layout(char const * layoutIn);
    Layout(gna_tensor_order order);
    ~Layout() = default;

    operator gna_tensor_order() const;

    void ValidateNumberOfDimensions(size_type shapeDimensions) const;

    void Reshape(Layout const & newLayout, size_type shapeDimensions);
    int32_t GetApiIndex(gna_tensor_dim dim) const;
    int32_t GetApiIndex(char dim) const;
private:
    static char const * GetOrderString(gna_tensor_order order);

    static const std::map<const std::string, gna_tensor_order> orderStrings;
    //gna_tensor_order OrderFromLayout() const;
};

}
