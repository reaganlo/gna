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
    static void ExpectGtZero(int64_t val, Gna2ItemType valType);
    static void ExpectEqual(int64_t val, int64_t ref, Gna2ItemType valType);
    static void ExpectBelowEq(int64_t val, int64_t ref, Gna2ItemType valType);
    static void ExpectAboveEq(int64_t val, int64_t ref, Gna2ItemType valType);
    static void ExpectMultiplicityOf(int64_t val, int64_t factor, Gna2ItemType valType);
    static void ExpectBufferNotNull(const void * const buffer, int32_t operandIndex = GNA2_DISABLED);
    static void ExpectBufferAligned(const void * const buffer, const uint32_t alignment);

    template<class A, class B>
    static void ExpectEqual(A val, B ref, Gna2ItemType valType)
    {
        ExpectEqual(static_cast<int64_t>(val), static_cast<int64_t>(ref), valType);
    }

    template<class A, class B>
    static void ExpectBelowEq(A val, B ref, Gna2ItemType valType)
    {
        ExpectBelowEq(static_cast<int64_t>(val), static_cast<int64_t>(ref), valType);
    }

    template<class A, class B>
    static void ExpectAboveEq(A val, B ref, Gna2ItemType valType)
    {
        ExpectAboveEq(static_cast<int64_t>(val), static_cast<int64_t>(ref), valType);
    }

    template<class A, class B>
    static void ExpectMultiplicityOf(A val, B factor, Gna2ItemType valType)
    {
        ExpectMultiplicityOf(static_cast<int64_t>(val), static_cast<int64_t>(factor), valType);
    }

    static void SaveLastError(const Gna2ModelError& modelError);
    static void PopLastError(Gna2ModelError& error);
    static Gna2Status ExecuteSafelyAndStoreLastError(const std::function<Gna2Status()>& commandIn);

    static Gna2ModelError GetCleanedError();
    static Gna2ModelError GetStatusError(Gna2Status status);

    static void SetOperandIndexRethrow(GnaException& e, uint32_t index);
private:
    static Gna2ModelError lastError;
};

class GnaModelErrorException : public GnaException
{
public:
    GnaModelErrorException(Gna2ModelError errorIn = ModelErrorHelper::GetCleanedError()) :
        GnaException{ Gna2StatusModelConfigurationInvalid },
        error{ errorIn }
    {
    }
    GnaModelErrorException(const GnaException& e) :
        GnaException(e),
        error{ ModelErrorHelper::GetStatusError(e.GetStatus()) }
    {
    }
    GnaModelErrorException(uint32_t layerIndex, Gna2Status code = Gna2StatusUnknownError);
    GnaModelErrorException(Gna2ItemType item, Gna2ErrorType errorType, int64_t value);

    Gna2ModelError GetModelError() const
    {
        return error;
    }
    void SetLayerIndex(uint32_t index)
    {
        error.Source.OperationIndex = static_cast<int32_t>(index);
    }
    void SetOperandIndex(uint32_t operandIndex)
    {
        error.Source.OperandIndex = static_cast<int32_t>(operandIndex);
    }
    void SetDimensionIndex(int32_t dimensionIndex)
    {
        error.Source.ShapeDimensionIndex = dimensionIndex;
    }

    virtual ~GnaModelErrorException() = default;
private:
    Gna2ModelError error;
};

}