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

#include "Capabilities.h"

#include "GnaException.h"
#include "Validator.h"
#include "Component.h"

using namespace GNA;

gna_tensor_order FullCapabilitiesMap::GetOrder(const LayerValidator& validator) const
{
    return GetLatestCaps(validator)->Order.Value;
}

ComponentLimits * FullCapabilitiesMap::GetLatestCaps(const LayerValidator& validator) const
{
    try
    {
        auto& caps = at(validator.Operation);
        for (auto latestHW = caps.rbegin(); latestHW != caps.rend(); ++latestHW)
        {
            if (latestHW->first <= validator.Device)
            {
                return latestHW->second.get();
            }
        }
        throw GnaException(XNN_ERR_LYR_CFG);
    }
    // operation or device not supported at all
    catch (const std::out_of_range&)
    {
        throw GnaException(XNN_ERR_LYR_CFG);
    }
}
