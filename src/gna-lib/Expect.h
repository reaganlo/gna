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

#include "ParameterLimits.h"
#include "Shape.h"

#include <vector>
#include <functional>

namespace GNA
{

class Memory;

// Validator utility
class Expect
{
public:
    // If condition is NOT satisfied prints error status code and throws exception .
    inline static void True(const bool condition, const Gna2Status error)
    {
        if (!condition)
        {
            throw GnaException(error);
        }
    }

    // If condition is satisfied prints error status code and throws exception .
    inline static void False(const bool condition, const Gna2Status error)
    {
        True(!condition, error);
    }

    template<typename T>
    inline static void Equal(const T a, const T b, const Gna2Status error)
    {
        True(a == b, error);
    }

    template<typename T>
    inline static void One(const T value, const Gna2Status error)
    {
        True(value == static_cast<T>(1), error);
    }

    template<typename T>
    inline static void Zero(const T value, const Gna2Status error)
    {
        True(value == static_cast<T>(0), error);
    }

    // If value is not > 0 prints error status code and throws exception .
    template<typename T>
    inline static void GtZero(const T value, const Gna2Status error)
    {
        True(value > static_cast<T>(0), error);
    }

    inline static void Success(const Gna2Status status)
    {
        True(Gna2StatusSuccess == status, status);
    }

    // If pointer is nullptr prints error status code and throws exception.
    inline static void NotNull(const void* pointer, const Gna2Status error)
    {
        True(nullptr != pointer, error);
    }

    // If pointer is nullptr prints error status code and throws exception.
    inline static void NotNull(const void* pointer)
    {
        NotNull(pointer, Gna2StatusNullArgumentNotAllowed);
    }

    // If pointer is NOT nullptr prints error status code and throws exception.
    inline static void Null(const void* pointer)
    {
        True(nullptr == pointer, Gna2StatusNullArgumentRequired);
    }

    // If pointer is nullptr prints error status code and throws exception.
    template<typename T>
    inline static void NotNull(std::unique_ptr<T> const & pointer)
    {
        True(pointer.operator bool(), Gna2StatusNullArgumentNotAllowed);
    }

    // If pointer is not aligned to alignment prints error status code and throws exception.
    inline static void AlignedTo(const void* pointer, const AlignLimits& alignLimits)
    {
        True(0 == ((reinterpret_cast<uintptr_t>(pointer)) % alignLimits.Value), alignLimits.Error);
    }

    // If pointer is not aligned to alignment prints error status code and throws exception.
    inline static void AlignedTo(const void* pointer, const uint32_t alignment = GNA_MEM_ALIGN)
    {
        AlignedTo(pointer, {alignment, Gna2StatusMemoryAlignmentInvalid});
    }

    // If pointer is not 64 B aligned prints error status code and throws exception.
    inline static void ValidBuffer(const void* pointer, const uint32_t alignment = GNA_MEM_ALIGN)
    {
        NotNull(pointer);
        AlignedTo(pointer, alignment);
    }

    // If pointer is not 64 B aligned prints error status code and throws exception.
    inline static void ValidBuffer(const void* pointer, const AlignLimits& alignLimits)
    {
        NotNull(pointer);
        AlignedTo(pointer, alignLimits);
    }

    inline static bool InMemoryRange(
        const void* buffer, const size_t bufferSize,
        const void *memory, const size_t memorySize)
    {
        auto *bufferEnd = static_cast<const uint8_t*>(buffer) + bufferSize;
        auto *memoryEnd = static_cast<const uint8_t*>(memory) + memorySize;

        return (buffer >= memory) && (bufferEnd <= memoryEnd);
    }

    // If pointers do not fit in user memory, throws Gna2StatusXnnErrorInvalidBuffer error
    inline static void ValidBoundaries(
        const void *buffer, const size_t bufferSize,
        const void *memory, const size_t memorySize)
    {
        False(InMemoryRange(buffer, bufferSize, memory, memorySize), Gna2StatusXnnErrorInvalidBuffer);
    }

    // If parameter is not multiplicity of multiplicity prints error status code and throws exception.
    template<typename T>
    inline static void MultiplicityOf(const T parameter, const T multiplicity,
        const Gna2Status error)
    {
        True(0 == (parameter % multiplicity), error);
    }

    // If parameter is not multiplicity of multiplicity prints error status code and throws exception.
    template<typename T>
    inline static void MultiplicityOf(const T parameter, const T multiplicity)
    {
        MultiplicityOf(parameter, multiplicity, Gna2StatusNotMultipleOf);
    }

    // If parameter is not multiplicity of multiplicity prints error status code and throws exception.
    template<typename T>
    inline static void MultiplicityOf(const T parameter, const ValueLimits<T>& multiplicity)
    {
        MultiplicityOf(parameter, multiplicity.Value, multiplicity.Error);
    }

    // If parameter is not in range of <0, b> prints error status code and throws exception.
    template<typename T = uint32_t>
    inline static void InRange(const T parameter, const T max,
        const Gna2Status error)
    {
        InRange(parameter, static_cast<T>(0), max, error);
    }

    // If parameter is not in range of <a, b> prints error status code and throws exception.
    template<typename T = uint32_t>
    inline static void InRange(const T parameter, const T a, const T b,
        const Gna2Status error)
    {
        False(parameter < a, error);
        False(parameter > b, error);
    }

    // If parameter is not in range of <a, b> prints error status code and throws exception.
    template<typename T = uint32_t>
    inline static void InRange(const T parameter, const T a, const T b,
        const Gna2Status lowerThanError, const Gna2Status greaterThanError)
    {
        False(parameter < a, lowerThanError);
        False(parameter > b, greaterThanError);
    }

    // If parameter is not in range of <a, b> prints error status code and throws exception.
    template<typename T = uint32_t>
    inline static void InRange(const T& parameter, const RangeLimits<T>& limit)
    {
        InRange(parameter, limit.Min.Value, limit.Max.Value, limit.Min.Error, limit.Max.Error);
    }

    // If parameter is not in set prints error status code and throws exception.
    template<typename T>
    static void InSet(const T& parameter, const SetLimits<T>& setLimits)
    {
        InSet(parameter, setLimits, setLimits.Error);
    }

    // If parameter is not in set prints error status code and throws exception.
    template<typename T, typename S> static void InSet(const T& parameter, const std::vector<T>& setLimits,
        const S error)
    {
        for (const T& item : setLimits)
        {
            if (item == parameter)
            {
                return;
            }
        }
        throw GnaException(error);
    }

    // If any dimension in map is invalid prints error status code and throws exception.
    template<typename T>
    inline static void DimensionIsValid(const T& dimension, const RangeLimits<T>& limits)
    {
        Expect::InRange(dimension, limits);
        Expect::MultiplicityOf(dimension, {limits.Multipliers.GetEffective(), limits.Multipliers.Error});
    }

    // If any dimension in map is invalid prints error status code and throws exception.
    inline static void ShapeIsValid(const Shape& dimensions,
        const ShapeLimits& limits)
    {
        for (const auto& dim : dimensions)
        {
            try
            {
                auto limit = limits.at(dim.first);
                Expect::DimensionIsValid(dim.second, limit);
            }
            catch (GnaException& e)
            {
                throw GnaTensorException{e, dim.first};
            }
        }
    }

    // If any dimension in a is greater than in b prints error status code and throws exception.
    inline static void Compare(const Shape& a, const Shape& b, std::function<void(uint32_t,uint32_t)>&& comapre)
    {
        for (const auto& dimA : a)
        {
            try
            {
                auto dimB = b.at(dimA.first);
                comapre(dimA.second, dimB);
            }
            catch (GnaException& e)
            {
                throw GnaTensorException{e, dimA.first};
            }
        }
    }

    // If any dimension in a is different than in b prints error status code and throws exception.
    inline static void ShapesAreEqual(const Shape& a, const Shape& b)
    {
        Compare(a, b, [](uint32_t x, uint32_t y) {Expect::True(x == y, Gna2StatusXnnErrorLyrInvalidTensorDimensions);});
    }

    // If any dimension in a is greater than in b prints error status code and throws exception.
    inline static void Fits(const Shape& a, const Shape& b)
    {
        Compare(a, b, [](uint32_t x, uint32_t y) {Expect::True(x <= y, Gna2StatusXnnErrorLyrInvalidTensorDimensions);});
    }

protected:
    Expect() = delete;
    Expect(const Expect &) = delete;
    Expect& operator=(const Expect&) = delete;
};

}
