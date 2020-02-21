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

#include "ModelError.h"

#include "ApiWrapper.h"
#include "Expect.h"

using namespace GNA;

Gna2ModelError ModelErrorHelper::lastError = ModelErrorHelper::GetCleanedError();

void ModelErrorHelper::ExpectTrue(bool val, Gna2ModelError error)
{
    if (!val)
    {
        throw GnaModelErrorException(error);
    }
}

void ModelErrorHelper::ExpectGtZero(uint32_t val, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeNotGtZero;
    ExpectTrue(val > 0, e);
}

void ModelErrorHelper::SaveLastError(const Gna2ModelError& modelError)
{
    lastError = modelError;
}

void ModelErrorHelper::PopLastError(Gna2ModelError& error)
{
    Expect::True(lastError.Source.Type != Gna2ItemTypeNone, Gna2StatusUnknownError);
    error = lastError;
    lastError = GetCleanedError();
}

Gna2Status ModelErrorHelper::ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn)
{
    const std::function<ApiStatus()> command = [&]()
    {
        try
        {
            return commandIn();
        }
        catch (GnaModelErrorException& exception)
        {
            ModelErrorHelper::SaveLastError(exception.GetModelError());
            throw;
        }
    };
    return ApiWrapper::ExecuteSafely(command);
}

Gna2ModelError ModelErrorHelper::GetCleanedError()
{
    Gna2ModelError e = {};
    e.Reason = Gna2ErrorTypeNone;
    e.Value = 0;
    e.Source.Type = Gna2ItemTypeNone;
    e.Source.OperationIndex = GNA2_DISABLED;
    e.Source.OperandIndex = GNA2_DISABLED;
    e.Source.ParameterIndex = GNA2_DISABLED;
    e.Source.ShapeDimensionIndex = GNA2_DISABLED;
    for (unsigned i = 0; i < sizeof(e.Source.Properties) / sizeof(e.Source.Properties[0]); i++)
    {
        e.Source.Properties[0] = GNA2_DISABLED;
    }
    return e;
}

GnaModelErrorException GnaModelErrorException::CreateFromLayerAndCode(uint32_t layerIndex, Gna2Status code)
{
    auto e = ModelErrorHelper::GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Source.OperationIndex = static_cast<int32_t>(layerIndex);
    e.Value = code;
    return GnaModelErrorException{e};
}

GnaModelErrorException GnaModelErrorException::CreateFromLayerUnknown(uint32_t layerIndex)
{
    return CreateFromLayerAndCode(layerIndex, Gna2StatusUnknownError);
}
