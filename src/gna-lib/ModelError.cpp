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

void ModelErrorHelper::ExpectGtZero(int64_t val, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeNotGtZero;
    ExpectTrue(val > 0, e);
}

void ModelErrorHelper::ExpectEqual(int64_t val, int64_t ref, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeNotEqual;
    ExpectTrue(val == ref, e);
}

void ModelErrorHelper::ExpectBelowEq(int64_t val, int64_t ref, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeAboveRange;
    ExpectTrue(val <= ref, e);
}

void ModelErrorHelper::ExpectAboveEq(int64_t val, int64_t ref, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeBelowRange;
    ExpectTrue(val >= ref, e);
}

void ModelErrorHelper::ExpectMultiplicityOf(int64_t val, int64_t factor, Gna2ItemType valType)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = valType;
    e.Value = val;
    e.Reason = Gna2ErrorTypeNotMultiplicity;
    ExpectTrue(val == 0 || (factor != 0 && (val % factor) == 0), e);
}

void ModelErrorHelper::ExpectBufferNotNull(const void * const buffer, int32_t operandIndex)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeOperandData;
    e.Source.OperandIndex = operandIndex;
    e.Value = 0;
    e.Reason = Gna2ErrorTypeNullNotAllowed;
    ExpectTrue(buffer != nullptr, e);
}

void ModelErrorHelper::ExpectBufferAligned(const void * const buffer, const uint32_t alignment)
{
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeOperandData;
    e.Value = reinterpret_cast<int64_t>(buffer);
    e.Reason = Gna2ErrorTypeNotAligned;
    ExpectTrue(alignment != 0 && ((e.Value % alignment) == 0), e);
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
            throw GnaException(Gna2StatusModelConfigurationInvalid);
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

Gna2ModelError ModelErrorHelper::GetStatusError(Gna2Status status)
{
    auto e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Reason = Gna2ErrorTypeOther;
    e.Value = status;
    return e;
}

GnaModelErrorException::GnaModelErrorException(
    Gna2ItemType item,
    Gna2ErrorType errorType,
    int64_t value)
    : GnaException{ Gna2StatusModelConfigurationInvalid }
{
    error = ModelErrorHelper::GetCleanedError();
    error.Source.Type = item;
    error.Reason = errorType;
    error.Value = value;
}

GnaModelErrorException::GnaModelErrorException(uint32_t layerIndex, Gna2Status capturedCode)
    : GnaException{ Gna2StatusModelConfigurationInvalid }
{
    error = ModelErrorHelper::GetStatusError(capturedCode);
    SetLayerIndex(layerIndex);
}

void ModelErrorHelper::SetOperandIndexRethrow(GnaException& e, uint32_t index)
{
    auto x = dynamic_cast<GnaModelErrorException*>(&e);
    if(x != nullptr)
    {
        x->SetOperandIndex(index);
        throw;
    }
    GnaModelErrorException n(e);
    n.SetOperandIndex(index);
    throw n;
}
