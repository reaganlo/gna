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

void igemv16(
    const uint32_t M,
    const uint32_t K,
    const int16_t* I,
    const int16_t *FB,
    const int16_t* W,
    const nn_bias_s *B,
          int32_t *O, 
          uint32_t *nSat)
{
    int16_t *input;
    int16_t *feedback;
    int16_t *feed_end = const_cast<int16_t*>(FB)+M;

    nn_bias_s *bias = const_cast<nn_bias_s*>(B), *bias_end = bias + M;
    int32_t *out = const_cast<int32_t*>(O);
    int16_t *weight = const_cast<int16_t*>(W);

    __m128i in, w, ma;
    __m128i inm0, inm1, inm2, inm3;
    __m128i zero;
    __m128i acc;

    zero = _mm_setzero_si128();

    uint32_t allElems = K + M; // total # of in + out/fb elements
    uint32_t i, j, k, kk;
    int64_t sum;

    uint32_t KK = K - K % VEC_16CAP;
    uint32_t part_sz = hw_buf_size[0];
    uint32_t kpart_sz = K % part_sz;
    uint32_t mpart_sz = M < part_sz - kpart_sz ? M
        : part_sz - kpart_sz;
    uint32_t mm = mpart_sz - mpart_sz % VEC_16CAP;
    uint32_t MM = M - (M - mpart_sz) % VEC_16CAP;

    uint32_t kparts = K / part_sz;
    uint32_t mparts = (M - mpart_sz) / part_sz;

    acc = _mm_setzero_si128();

    for (; bias < bias_end; bias++)
    {
        input = const_cast<int16_t*>(I);
        feedback = const_cast<int16_t*>(FB);
        sum = *bias;

        // compute parts using SSE 
        // if K has modulo 16 remainder, leave it
        for (j = 0; j < kparts + 1; j++)
        {
            in = _mm_lddqu_si128((__m128i*)input);
            w = _mm_lddqu_si128((__m128i*)weight);
            for (k = 0; k < part_sz && (j*part_sz + k < KK); k += VEC_16CAP)
            {
                input += VEC_16CAP;
                weight += VEC_16CAP;

                ma = _mm_madd_epi16(in, w);
                inm0 = _mm_cvtepi32_epi64(ma);
                inm1 = _mm_cvtepi32_epi64(_mm_bsrli_si128(ma, 8));
                inm2 = _mm_add_epi64(inm0, inm1);

                acc = _mm_add_epi64(acc, inm2);

                in = _mm_lddqu_si128((__m128i*)input);
                w = _mm_lddqu_si128((__m128i*)weight);
            }

            // saturate if part size achieved
            if (k == part_sz)
            {
                sum += vec_sum(acc);
                acc = _mm_setzero_si128();
                saturate_store_out(&sum, out, nSat);
                sum = (int64_t)*out;
            }
        }

        // compute remainder
        for (k = KK; k < K; k++)
        {
            sum += *input++ * *weight++;
        }

        in = _mm_lddqu_si128((__m128i*)feedback);
        w = _mm_lddqu_si128((__m128i*)weight);

        // compute using SSE instructions until additions reach part size
        // or if loop reaches end of M (without the modulo 16 remainder)
        for (k = 0; k < mm; k += VEC_16CAP)
        {
            feedback += VEC_16CAP;
            weight += VEC_16CAP;

            in = _mm_madd_epi16(in, w);
            inm0 = _mm_cvtepi32_epi64(in);
            inm1 = _mm_cvtepi32_epi64(_mm_bsrli_si128(in, 8));
            inm2 = _mm_add_epi64(inm0, inm1);

            acc = _mm_add_epi64(acc, inm2);

            in = _mm_lddqu_si128((__m128i*)feedback);
            w = _mm_lddqu_si128((__m128i*)weight);
        }

        // if part size wasn't reached, but there is still M remainder
        for (; k < mpart_sz; k++)
        {
            sum += (int32_t)(*feedback++ * *weight++);
        }

        sum += vec_sum(acc);
        acc = _mm_setzero_si128();
        saturate_store_out(&sum, out, nSat);
        sum = (int64_t)*out;

        for (j = 0; j < mparts + 1; j++)
        {
            for (kk = 0; kk < part_sz && (j*part_sz + mpart_sz + kk < MM); kk += VEC_16CAP)
            {
                feedback += VEC_16CAP;
                weight += VEC_16CAP;

                in = _mm_madd_epi16(in, w);
                inm0 = _mm_cvtepi32_epi64(in);
                inm1 = _mm_cvtepi32_epi64(_mm_bsrli_si128(in, 8));
                inm2 = _mm_add_epi64(inm0, inm1);

                acc = _mm_add_epi64(acc, inm2);

                in = _mm_lddqu_si128((__m128i*)feedback);
                w = _mm_lddqu_si128((__m128i*)weight);
            }

            if (kk == part_sz)
            {
                sum += vec_sum(acc);
                acc = _mm_setzero_si128();
                saturate_store_out(&sum, out, nSat);
                sum = (int64_t)*out;
            }
        }

        // if there's remainder from mparts
        for (; feedback < feed_end;)
        {
            sum += (int32_t)(*feedback++ * *weight++);
        }

        sum += vec_sum(acc);
        acc = _mm_setzero_si128();
        saturate_store_out(&sum, out, nSat);
        sum = (int64_t)*out;

        out++;
    }
}