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
class Expect
{
public:
    // If condition is NOT satisfied prints error status code and throws exception .
    inline static void True(const bool condition, const status_t status)
    {
        if (!condition)
        {
            throw GnaException(status);
        }
    }

    // If condition is satisfied prints error status code and throws exception .
    inline static void False(const bool condition, const status_t status)
    {
        True(!condition, status);
    }

    // If pointer is nullptr prints error status code and throws exception.
    inline static void NotNull(const void* pointer, const status_t status)
    {
        True(nullptr != pointer, status);
    }

    // If pointer is nullptr prints error status code and throws exception.
    inline static void NotNull(const void* pointer)
    {
        NotNull(pointer, GNA_NULLARGNOTALLOWED);
    }

    // If pointer is NOT nullptr prints error status code and throws exception.
    inline static void Null(const void* pointer)
    {
        True(nullptr == pointer, GNA_NULLARGREQUIRED);
    }

    // If pointer is not aligned to alignment prints error status code and throws exception.
    inline static void AlignedTo(const void* pointer, const uint32_t alignment, const status_t status)
    {
        True(0 == (((uintptr_t)pointer) % alignment), status);
    }

    // If pointer is not aligned to alignment prints error status code and throws exception.
    inline static void AlignedTo(const void* pointer, const uint32_t alignment)
    {
        AlignedTo(pointer, alignment, GNA_BADMEMALIGN);
    }

    // If pointer is not 64 B aligned prints error status code and throws exception.
    inline static void AlignedTo64(const void* pointer)
    {
        AlignedTo(pointer, 64);
    }

    // If pointer is not 64 B aligned prints error status code and throws exception.
    inline static void ValidBuffer(const void* pointer)
    {
        NotNull(pointer);
        AlignedTo64(pointer);
    }

    // If pointer is not 64 B aligned prints error status code and throws exception.
    inline static void ValidBuffer(const void* pointer, const status_t status)
    {
        NotNull(pointer, status);
        AlignedTo(pointer, 64, status);
    }

    // If parameter is not multiplicity of multiplicity prints error status code and throws exception.
    inline static void MultiplicityOf(const uint32_t parameter, const uint32_t multiplicity)
    {
        True(0 == (parameter % multiplicity), GNA_ERR_NOT_MULTIPLE);
    }

    // If parameter is not in range of <a, b> prints error status code and throws exception.
    inline static void InRange(const size_t parameter, const size_t a, const size_t b,
        const status_t status)
    {
        False(parameter < a, status);
        False(parameter > b, status);
    }

protected:
    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    Expect() = delete;
    Expect(const Expect &) = delete;
    Expect& operator=(const Expect&) = delete;
};

}
