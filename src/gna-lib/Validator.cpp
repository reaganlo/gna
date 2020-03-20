/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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
#include "Expect.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include "common.h"
#include "gna-api.h"

#include <cstddef>

using namespace GNA;

BaseValidator::BaseValidator(
    const HardwareCapabilities hwCapabilities,
    const ValidBoundariesFunctor validBoundariesIn) :
    HwCapabilities{ hwCapabilities },
    bufferValidator{ validBoundariesIn }
{
}

void BaseValidator::ValidateBuffer(const void * const buffer, size_t size, const uint32_t alignment) const
{
    ModelErrorHelper::ExpectNotNull(buffer);
    ModelErrorHelper::ExpectBufferAligned(buffer, alignment);
    bufferValidator(buffer, size);
}

LayerValidator::LayerValidator(const BaseValidator& validator, nn_operation operation) :
    BaseValidator{ validator },
    Operation{ operation }
{
}

Validator::Validator(const LayerValidator & validator, const FullCapabilitiesMap & capabilities) :
    LayerValidator{ validator },
    Capabilities{ capabilities.GetLatestCaps(validator) },
    Order{ Capabilities->Order.Value }
{
}
