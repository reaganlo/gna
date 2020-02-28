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

#include "Shape.h"
#include "Tensor.h"

#include "common.h"
#include "gna-api-types-xnn.h"

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;

struct LayerInput : public Tensor
{
    LayerInput(const nn_layer &layer, const LayerValidator& validatorIn);
    LayerInput(const Gna2Operation &operation, const LayerValidator& validatorIn);
    virtual ~LayerInput() = default;

    static bool IsInputInterleave(const Gna2Tensor &apiTensor,
                       const BaseValidator& validatorIn);

    const uint32_t Grouping;
    const uint32_t ElementCount;

protected:
    static const FullCapabilitiesMap capabilities;

    static Shape GetDimensions(const nn_layer& layer, gna_tensor_order order);

    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const override;
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(const nn_layer& layer) const override;
};

}
