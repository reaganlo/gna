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

/** @brief Add 32b signed integers inside 256b register */
static inline int32_t _mm256_hsum_epi32(__m256i acc0)
{
    __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(acc0, 1), _mm256_castsi256_si128(acc0));
    __m128i sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    __m128i sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 1));

    return _mm_cvtsi128_si32(sum32);
}
