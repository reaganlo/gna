/*
INTEL CONFIDENTIAL
Copyright 2020 Intel Corporation.

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

#include "GnaException.h"

#include "gna2-common-api.h"
#include "gna2-model-api.h"

#include <functional>

namespace GNA
{

class ModelErrorHelper
{
public:
    static void ExpectTrue(bool val, Gna2ModelError error);
    static void ExpectGtZero(uint32_t val, Gna2ItemType valType);
    static void SaveLastError(const Gna2ModelError& modelError);
    static void PopLastError(Gna2ModelError& error);
    static Gna2Status ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn);

    static Gna2ModelError GetCleanedError();


private:
    static Gna2ModelError lastError;
};

class GnaModelErrorException : public GnaException
{
public:
    GnaModelErrorException(Gna2ModelError errorIn) :
        GnaException{ Gna2StatusModelConfigurationInvalid },
        error{ errorIn }
    {
    }
    Gna2ModelError GetModelError() const
    {
        return error;
    }
    static GnaModelErrorException CreateFromLayerAndCode(uint32_t layerIndex, Gna2Status code);
    static GnaModelErrorException CreateFromLayerUnknown(uint32_t layerIndex);
    virtual ~GnaModelErrorException() = default;
private:
    Gna2ModelError error;
};

}