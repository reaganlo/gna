/*
 INTEL CONFIDENTIAL
 Copyright 2021 Intel Corporation.

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
#include <cstdint>
#include <immintrin.h>
#include <limits>

/** @brief Add 32b signed integers inside 256b register */
static inline int32_t _mm256_hsum_epi32(__m256i acc0)
{
    __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(acc0, 1), _mm256_castsi256_si128(acc0));
    __m128i sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    __m128i sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 1));

    return _mm_cvtsi128_si32(sum32);
}

/** @brief Add 64b signed integers inside 256b register */
static inline int64_t _mm256_hsum_epi64(__m256i acc0)
{
    __m128i sum128 = _mm_add_epi64(_mm256_extracti128_si256(acc0, 1), _mm256_castsi256_si128(acc0));
    __m128i sum64 = _mm_add_epi64(sum128, _mm_unpackhi_epi64(sum128, sum128));

    return _mm_cvtsi128_si64(sum64);
}

/** @brief Add 32b signed integers using saturation inside 256b register;
 *  set MSB of given 32b integer if saturation occured for given pair. */
static inline __m256i _mm256_adds_epi32(__m256i a, __m256i b, __m256i *setHighBitOnSat)
{
    static const __m256i satMax = _mm256_set1_epi32((std::numeric_limits<int32_t>::max)());
    __m256i sum = _mm256_add_epi32(a, b);
    __m256i ov = _mm256_andnot_si256(_mm256_xor_si256(b, a), _mm256_xor_si256(b, sum));
    a = _mm256_xor_si256(satMax, _mm256_srai_epi32(a, 32));
    *setHighBitOnSat = _mm256_or_si256(ov, *setHighBitOnSat);
    return _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(sum), _mm256_castsi256_ps(a), _mm256_castsi256_ps(ov)));
}

/** @brief Check if any MSB of 32b integers is set inside 256b register */
static inline bool _mm256_test_anyMSB_epi32(__m256i a)
{
    return 0 != _mm256_movemask_ps(_mm256_castsi256_ps(a));
}

/** @brief Check if any bit of 256b register is set */
static inline bool _mm256_test_any(__m256i a)
{
    return 0 == _mm256_testc_si256(a, _mm256_set1_epi64x(-1));
}
