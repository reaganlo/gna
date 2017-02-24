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

#include "igemv.h"
#include "igemv16.h"

void 
igemv16(
    const uint32_t M,
    const uint32_t K,
    const int16_t* I,
    const int16_t *FB,
    const int16_t* W,
    const nn_bias_s *B,
          int32_t *Y,
          uint32_t *nSat)
{
    uint32_t LDA = M + K;
    int16_t *input = const_cast<int16_t*>(I);
    int16_t *feedback = const_cast<int16_t*>(FB);

    int16_t *ie = input + K - K % 8;
    int16_t *fe = feedback + M - M % 8;

    nn_bias_s *b = const_cast<nn_bias_s*>(B), *be = b + M;
    int32_t *y = const_cast<int32_t*>(Y);
    int16_t *w0 = const_cast<int16_t*>(W);
    int16_t *w1 = w0 + K;

    __m128i v0, v1, v2, v3, v4, v5, v6, v7, v8;

    for (; b < be; b++)
    {
        v2 = _mm_setzero_si128();

        input = const_cast<int16_t*>(I);
        feedback = const_cast<int16_t*>(FB);

        v0 = _mm_lddqu_si128((__m128i*)input);
        v1 = _mm_lddqu_si128((__m128i*)w0);

        while (input < ie)
        {
            input += 8;
            w0 += 8;

            v1 = _mm_madd_epi16(v0, v1);
            v2 = _mm_add_epi32(v1, v2);

            v0 = _mm_lddqu_si128((__m128i*)input);
            v1 = _mm_lddqu_si128((__m128i*)w0);
        }

        v0 = _mm_lddqu_si128((__m128i*)feedback);
        v1 = _mm_lddqu_si128((__m128i*)w1);

        while (feedback < fe)
        {
            feedback += 8;
            w1 += 8;

            v1 = _mm_madd_epi16(v0, v1);
            v2 = _mm_add_epi32(v1, v2);

            v0 = _mm_lddqu_si128((__m128i*)feedback);
            v1 = _mm_lddqu_si128((__m128i*)w1);
        }

        *y = vec_sum(v2) + *b;

        while (input < ie + K % 8)
        {
            *y += *input++ * *w0++;
        }

        while (feedback < fe + M % 8)
        {
            *y += *feedback++ * *w1++;
        }

        y++;

        w0 += LDA - K;
        w1 += LDA - M;
    }
}