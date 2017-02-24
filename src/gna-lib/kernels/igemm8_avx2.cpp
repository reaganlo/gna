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
#include "igemv8.h"
#include "string.h"
void igemm8(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int8_t *W,
    const nn_bias_c *B,
          int32_t *O,
          uint32_t *nSat,
          aligned_fv_bufs *bufs)
{
    uint32_t i, j, ix, ix_end;
    uint32_t KT = K % VEC_16CAP;
    uint32_t KK = K - KT;
    ix_end = KK / VEC_16CAP;
    int8_t *weight = const_cast<int8_t*>(W);
    int32_t *y = O;
    nn_bias_c *b, *be = const_cast<nn_bias_c*>(B)+M;

    int16_t *input_0, *input_1, *input_2, *input_3, *input_4, *input_5, *input_6, *input_7;
    __m256i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    __m256i in0, in1, in2, in3, in4, in5, in6, in7, w;
    __m256i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    __m128i s1, s2, s3, s4;

    if (1 == N)
    {
        input_0 = const_cast<int16_t*>(I)+KK;
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            in_ptr0 = (__m256i*)I;
            acc0 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                acc0 = _mm256_add_epi32(acc0, in0);
            }

            *y = b->bias + vec_sum(acc0) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                *y += input_0[j] * *weight * b->multiplier;
            }

            y++;
        }
        return;
    }

    switch (N)
    {
    case 8: 
        for (i = 0; i < K; i++) bufs->d7[i] = I[i*N + 7];
        input_7 = bufs->d7 + KK;
        in_ptr7 = (__m256i*)bufs->d7;
    case 7: 
        for (i = 0; i < K; i++) bufs->d6[i] = I[i*N + 6];
        input_6 = bufs->d6 + KK;
        in_ptr6 = (__m256i*)bufs->d6;
    case 6: 
        for (i = 0; i < K; i++) bufs->d5[i] = I[i*N + 5];
        input_5 = bufs->d5 + KK;
        in_ptr5 = (__m256i*)bufs->d5;
    case 5: 
        for (i = 0; i < K; i++) bufs->d4[i] = I[i*N + 4];
        input_4 = bufs->d4 + KK;
        in_ptr4 = (__m256i*)bufs->d4;
    case 4: 
        for (i = 0; i < K; i++) bufs->d3[i] = I[i*N + 3];
        input_3 = bufs->d3 + KK;
        in_ptr3 = (__m256i*)bufs->d3;
    case 3: 
        for (i = 0; i < K; i++) bufs->d2[i] = I[i*N + 2];
        input_2 = bufs->d2 + KK;
        in_ptr2 = (__m256i*)bufs->d2;
    case 2: 
        for (i = 0; i < K; i++) bufs->d1[i] = I[i*N + 1];
        input_1 = bufs->d1 + KK;
        in_ptr1 = (__m256i*)bufs->d1;
        for (i = 0; i < K; i++) bufs->d0[i] = I[i*N];
        input_0 = bufs->d0 + KK;
        in_ptr0 = (__m256i*)bufs->d0;
    }

    if (2 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
            }

            y[0] = b->bias + vec_sum(acc0) * b->multiplier;
            y[1] = b->bias + vec_sum(acc1) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
    if (3 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
            }

            y[0] = b->bias + vec_sum(acc0) * b->multiplier;
            y[1] = b->bias + vec_sum(acc1) * b->multiplier;
            y[2] = b->bias + vec_sum(acc2) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
                y[2] += input_2[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
    if (4 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
            }

            y[0] = b->bias + vec_sum(acc0) * b->multiplier;
            y[1] = b->bias + vec_sum(acc1) * b->multiplier;
            y[2] = b->bias + vec_sum(acc2) * b->multiplier;
            y[3] = b->bias + vec_sum(acc3) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
                y[2] += input_2[j] * *weight * b->multiplier;
                y[3] += input_3[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
    if (5 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
            }

            y[0] = b->bias + vec_sum(acc0) * b->multiplier;
            y[1] = b->bias + vec_sum(acc1) * b->multiplier;
            y[2] = b->bias + vec_sum(acc2) * b->multiplier;
            y[3] = b->bias + vec_sum(acc3) * b->multiplier;
            y[4] = b->bias + vec_sum(acc4) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
                y[2] += input_2[j] * *weight * b->multiplier;
                y[3] += input_3[j] * *weight * b->multiplier;
                y[4] += input_4[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
    if (6 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
            }

            y[0] = b->bias + vec_sum(acc0) * b->multiplier;
            y[1] = b->bias + vec_sum(acc1) * b->multiplier;
            y[2] = b->bias + vec_sum(acc2) * b->multiplier;
            y[3] = b->bias + vec_sum(acc3) * b->multiplier;
            y[4] = b->bias + vec_sum(acc4) * b->multiplier;
            y[5] = b->bias + vec_sum(acc5) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
                y[2] += input_2[j] * *weight * b->multiplier;
                y[3] += input_3[j] * *weight * b->multiplier;
                y[4] += input_4[j] * *weight * b->multiplier;
                y[5] += input_5[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
    if (7 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);
                in6 = _mm256_load_si256(in_ptr6 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);
                in6 = _mm256_madd_epi16(in6, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
                acc6 = _mm256_add_epi32(acc6, in6);
            }

            y[0] = b->bias + vec_sum(acc0) * b->multiplier;
            y[1] = b->bias + vec_sum(acc1) * b->multiplier;
            y[2] = b->bias + vec_sum(acc2) * b->multiplier;
            y[3] = b->bias + vec_sum(acc3) * b->multiplier;
            y[4] = b->bias + vec_sum(acc4) * b->multiplier;
            y[5] = b->bias + vec_sum(acc5) * b->multiplier;
            y[6] = b->bias + vec_sum(acc6) * b->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
                y[2] += input_2[j] * *weight * b->multiplier;
                y[3] += input_3[j] * *weight * b->multiplier;
                y[4] += input_4[j] * *weight * b->multiplier;
                y[5] += input_5[j] * *weight * b->multiplier;
                y[6] += input_6[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
    if (8 == N)
    {
        for (b = const_cast<nn_bias_c*>(B); b < be; b++)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();
            acc7 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);
                in6 = _mm256_load_si256(in_ptr6 + ix);
                in7 = _mm256_load_si256(in_ptr7 + ix);

                w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);
                in6 = _mm256_madd_epi16(in6, w);
                in7 = _mm256_madd_epi16(in7, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
                acc6 = _mm256_add_epi32(acc6, in6);
                acc7 = _mm256_add_epi32(acc7, in7);
            }

            v0 = _mm256_set1_epi32(b->multiplier);
            v1 = _mm256_set1_epi32(b->bias);

            v2 = _mm256_hadd_epi32(acc0, acc1);
            v3 = _mm256_hadd_epi32(acc2, acc3);
            v4 = _mm256_hadd_epi32(v2, v3);

            v5 = _mm256_hadd_epi32(acc4, acc5);
            v6 = _mm256_hadd_epi32(acc6, acc7);
            v7 = _mm256_hadd_epi32(v5, v6);

            s1 = _mm256_castsi256_si128(v4);
            s2 = _mm256_extracti128_si256(v4, 1);

            s3 = _mm256_castsi256_si128(v7);
            s4 = _mm256_extracti128_si256(v7, 1);

            s1 = _mm_add_epi32(s1, s2);
            s2 = _mm_add_epi32(s3, s4);

            acc0 = _mm256_set_m128i(s2, s1);
            acc0 = _mm256_mullo_epi32(acc0, v0);
            acc0 = _mm256_add_epi32(acc0, v1);
            _mm256_store_si256((__m256i*)y, acc0);

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight * b->multiplier;
                y[1] += input_1[j] * *weight * b->multiplier;
                y[2] += input_2[j] * *weight * b->multiplier;
                y[3] += input_3[j] * *weight * b->multiplier;
                y[4] += input_4[j] * *weight * b->multiplier;
                y[5] += input_5[j] * *weight * b->multiplier;
                y[6] += input_6[j] * *weight * b->multiplier;
                y[7] += input_7[j] * *weight * b->multiplier;
            }

            y += N;
        }
        return;
    }
}