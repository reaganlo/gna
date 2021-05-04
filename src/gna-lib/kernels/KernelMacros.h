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

#include "gna2-inference-api.h"

#include <stdint.h>

#if !defined(_MSC_VER)
#include <immintrin.h>
#else
#include <intrin.h>
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define __forceinline inline
#endif

/**
* Macros for decoration of function names build with different optimizations
*/
#define PASTER(x,y)     x ## y
#define EVALUATOR(x,y)  PASTER(x,y)
#define KERNEL(NAME)    EVALUATOR(NAME, KERNEL_SUFFIX)

#define vec_accumulate KERNEL(vec_accumulate)
#define vec_sum KERNEL(vec_sum)
#define vec_sum32 KERNEL(vec_sum32)
#define vec_madd16 KERNEL(vec_madd16)

/**
 * Definitions acceleration/optimization macros
 *
 * * OPT_LEVEL      - Build acceleration/optimization mode for numerical comparison
 * * KERNEL_SUFFIX  - suffix for decorating kernel names build for each optimization
 */

#if !defined(__LP64__) && !defined(_WIN64)
#define _mm_extract_epi64(a, i) ((((int64_t)_mm_extract_epi32(a,i*2+1)<<32)|_mm_extract_epi32(a,i*2)))
#define _mm256_extract_epi64(a, i) ((((int64_t)_mm256_extract_epi32(a,i*2+1)<<32)|_mm256_extract_epi32(a,i*2)))
#endif

#if     defined(OPTGEN)

#define OPT_LEVEL       0
#define KERNEL_SUFFIX   _generic
constexpr auto KernelAcceleration = Gna2AccelerationModeGeneric;
constexpr auto HwConsistencyMode = false;

#elif   defined(OPTGEN_SAT)

#define OPT_LEVEL       1
#define KERNEL_SUFFIX   _generic_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeGeneric;
constexpr auto HwConsistencyMode = true;

#elif   defined(OPTSSE4)

#define OPT_LEVEL       2
#define KERNEL_SUFFIX   _sse4
constexpr auto KernelAcceleration = Gna2AccelerationModeSse4x2;
constexpr auto HwConsistencyMode = false;

__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi32(acc, x);
}
__forceinline int32_t vec_sum(__m128i x)
{
    return _mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}

#elif   defined(OPTSSE4_SAT)

#define OPT_LEVEL       3
#define KERNEL_SUFFIX   _sse4_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeSse4x2;
constexpr auto HwConsistencyMode = true;

__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi64(acc, _mm_add_epi64(
        _mm_cvtepi32_epi64(x),
        _mm_cvtepi32_epi64(_mm_srli_si128(x, 8))));
}
__forceinline int64_t vec_sum(__m128i x)
{
    return _mm_extract_epi64(x, 0) + _mm_extract_epi64(x, 1);
}
__forceinline int64_t vec_sum32(__m128i x)
{
    return (int64_t)_mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}

#elif   defined(OPTAVX1)

#define OPT_LEVEL       4
#define KERNEL_SUFFIX   _avx1
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx1;
constexpr auto HwConsistencyMode = false;

__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi32(acc, x);
}
__forceinline int32_t vec_sum(__m128i x)
{
    return _mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}
__forceinline __m128i vec_madd16(__m256i x, __m256i y)
{
    return _mm_add_epi32(
        _mm_madd_epi16(_mm256_castsi256_si128(x), _mm256_castsi256_si128(y)),
        _mm_madd_epi16(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1)));
}

#elif   defined(OPTAVX1_SAT)

#define OPT_LEVEL       5
#define KERNEL_SUFFIX   _avx1_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx1;
constexpr auto HwConsistencyMode = true;

__forceinline __m128i vec_accumulate(__m128i acc, __m128i x)
{
    return _mm_add_epi64(acc, x);
}
__forceinline int64_t vec_sum(__m128i x)
{
    return _mm_extract_epi64(x, 0) + _mm_extract_epi64(x, 1);
}
__forceinline __m128i vec_madd16(__m256i x, __m256i y)
{
    __m128i m0 = _mm_madd_epi16(_mm256_castsi256_si128(x), _mm256_castsi256_si128(y));
    __m128i m1 = _mm_madd_epi16(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1));
    return _mm_add_epi64(
        _mm_add_epi64(_mm_cvtepi32_epi64(m0), _mm_cvtepi32_epi64(_mm_srli_si128(m0, 8))),
        _mm_add_epi64(_mm_cvtepi32_epi64(m1), _mm_cvtepi32_epi64(_mm_srli_si128(m1, 8))));
}
__forceinline int64_t vec_sum32(__m128i x)
{
    return (int64_t)_mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1) + _mm_extract_epi32(x, 2) + _mm_extract_epi32(x, 3);
}

#elif   defined(OPTAVX2)

#define OPT_LEVEL       6
#define KERNEL_SUFFIX   _avx2
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx2;
constexpr auto HwConsistencyMode = false;

__forceinline __m256i vec_accumulate(__m256i acc, __m256i x)
{
    return _mm256_add_epi32(acc, x);
}
__forceinline int32_t vec_sum(__m256i x)
{
    return _mm256_extract_epi32(x, 0) + _mm256_extract_epi32(x, 1) + _mm256_extract_epi32(x, 2) + _mm256_extract_epi32(x, 3)
         + _mm256_extract_epi32(x, 4) + _mm256_extract_epi32(x, 5) + _mm256_extract_epi32(x, 6) + _mm256_extract_epi32(x, 7);
}

#elif   defined(OPTAVX2_SAT)

#define OPT_LEVEL       7
#define KERNEL_SUFFIX   _avx2_sat
constexpr auto KernelAcceleration = Gna2AccelerationModeAvx2;
constexpr auto HwConsistencyMode = true;

__forceinline __m256i vec_accumulate(__m256i acc, __m256i x)
{
    return _mm256_add_epi64(acc, _mm256_add_epi64(
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(x)),
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(x, 1))));
}
__forceinline int64_t vec_sum32(__m256i x)
{
    return (int64_t)_mm256_extract_epi32(x, 0) + _mm256_extract_epi32(x, 1) + _mm256_extract_epi32(x, 2) + _mm256_extract_epi32(x, 3)
         + _mm256_extract_epi32(x, 4) + _mm256_extract_epi32(x, 5) + _mm256_extract_epi32(x, 6) + _mm256_extract_epi32(x, 7);
}
__forceinline int64_t vec_sum(__m256i x)
{
    return (int64_t)_mm256_extract_epi64(x, 0) + _mm256_extract_epi64(x, 1) + _mm256_extract_epi64(x, 2) + _mm256_extract_epi64(x, 3);
}

#else

// Force compilation error to prevent build of unsupported acceleration mode
#error NO SUPPORTED ACCELERATION MODE DEFINED

#endif

// SSE4_2
#if OPT_LEVEL == 2 || OPT_LEVEL == 3
typedef __m128i* mm_ptr;
#define VEC_16CAP 8
__forceinline __m128i vec_madd16(__m128i x, __m128i y)
{
    return _mm_madd_epi16(x, y);
}

__forceinline __m128i vec_lddqu(void *ptr)
{
    return _mm_lddqu_si128((__m128i*)ptr);
}
__forceinline __m128i vec_load(void *ptr)
{
    return _mm_load_si128((__m128i*)ptr);
}
#endif

// SSE4_2 & AVX1
#if OPT_LEVEL > 1 && OPT_LEVEL < 6
typedef __m128i mm_vector;
__forceinline __m128i vec_setzero()
{
    return _mm_setzero_si128();
}
#endif

// AVX1+
#if OPT_LEVEL > 3

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

typedef __m256i* mm_ptr;
#define VEC_16CAP 16

__forceinline __m256i vec_lddqu(void *ptr)
{
    return _mm256_lddqu_si256((__m256i*)ptr);
}
__forceinline __m256i vec_load(void *ptr)
{
    return _mm256_load_si256((__m256i*)ptr);
}
#endif

// AVX2+
#if OPT_LEVEL > 5
typedef __m256i mm_vector;
__forceinline __m256i vec_madd16(__m256i x, __m256i y)
{
    return _mm256_madd_epi16(x, y);
}
__forceinline __m256i vec_setzero()
{
    return _mm256_setzero_si256();
}
#endif

#if OPT_LEVEL % 2 == 0
#define GNA_SAT 0
typedef int32_t gna_sum_t;
#else
#define GNA_SAT 1
typedef int64_t gna_sum_t;
#endif

#define SSE_16CAP 8

inline int32_t getBias(const void* ptr, uint32_t bytesPerElement, uint32_t idx = 0)
{
    switch (bytesPerElement)
    {
    case 1:
        return ((int8_t*)ptr)[idx];
    case 2:
        return ((int16_t*)ptr)[idx];
    case 4:
        return ((int32_t*)ptr)[idx];
    default:
        return 0;
    }
}