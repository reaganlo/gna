/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

namespace GNA
{

// Validator utility
class Validate
{
public:
    // TODO: rename - misleading name
    // If condition is satisfied prints error status code and throws exception .
    inline static void IsTrue(const bool condition, const status_t status)
    {
        if (condition)
        {
            ERR("FAILED with status: [%d]=%s\n", (int)status, GnaStatusToString(status));
            throw GnaException(status);
        }
    }

    // If pointer is nullptr prints error status code and throws exception.
    inline static void IsNull(const void* pointer)
    {
        IsTrue(nullptr == pointer, GNA_NULLARGNOTALLOWED);
    }

    // If pointer is not nullptr prints error status code and throws exception.
    inline static void IsNotNull(const void* pointer)
    {
        IsTrue(nullptr != pointer, GNA_NULLARGREQUIRED);
    }

    // If pointer is not aligned to alignment prints error status code and throws exception.
    inline static void IsAlignedTo(const void* pointer, const uint32_t alignment)
    {
        IsTrue(0 != (((uintptr_t)pointer) % alignment), GNA_BADMEMALIGN);
    }

    // If pointer is not 64 B aligned prints error status code and throws exception.
    inline static void IsAlignedTo64(const void* pointer)
    {
        IsAlignedTo(pointer, 64);
    }

    // If parameter is not multiplicity of multiplicity prints error status code and throws exception.
    inline static void IsMultiplicityOf(const uint32_t parameter, const uint32_t multiplicity)
    {
        IsTrue(0 != (parameter % multiplicity), GNA_ERR_NOT_MULTIPLE);
    }

    // If parameter is not in range of <a, b> prints error status code and throws exception.
    inline static void IsInRange(const uint32_t parameter, const uint32_t a, const uint32_t b,
        const status_t status)
    {
        IsTrue(parameter < a, status);
        IsTrue(parameter > b, status);
    }

protected:
    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    Validate() = delete;
    Validate(const Validate &) = delete;
    Validate& operator=(const Validate&) = delete;
};

}
