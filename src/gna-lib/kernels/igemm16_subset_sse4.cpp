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
#include <string.h>
void
igemm16_subset(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int16_t *W,
    const nn_bias_s *B,
          int32_t *Y,
    const uint32_t *AL,
    const uint32_t L,
          uint32_t *nSat,
          KernelBuffers* fvBuffers)
{
    int32_t *y = Y;
    int16_t *weight = const_cast<int16_t*>(W);
    int16_t *input_0, *input_1, *input_2, *input_3, *input_4, *input_5, *input_6, *input_7;

    uint32_t i, j, l, ix, ix_end;
    uint32_t KT = K % SSE_16CAP;
    uint32_t KK = K - KT;
    __m128i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    __m128i v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
    __m128i in0, in1, in2, in3, in4, in5, in6, in7, w;
    __m128i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m128i s1, s2, s3, s4;
    nn_bias_s *b, *be = const_cast<nn_bias_s*>(B)+M;

    if (1 == N)
    {
        in_ptr0 = (__m128i*)I;
        input_0 = const_cast<int16_t*>(I)+KK;
        ix_end = KK / SSE_16CAP;
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            *y = vec_sum(acc0) + *b;
            for (i = 0; i < KT; i++, weight++)
            {
                *y += input_0[i] * *weight;
            }
            y++;
        }

        return;
    }

    switch (N)
    {
    case 8: 
        for (i = 0; i < K; i++) fvBuffers->d7[i] = I[i*N + 7];
        in_ptr7 = (__m128i*)fvBuffers->d7;
        input_7 = fvBuffers->d7 + KK;
    case 7: 
        for (i = 0; i < K; i++) fvBuffers->d6[i] = I[i*N + 6];
        in_ptr6 = (__m128i*)fvBuffers->d6;
        input_6 = fvBuffers->d6 + KK;
    case 6: 
        for (i = 0; i < K; i++) fvBuffers->d5[i] = I[i*N + 5];
        in_ptr5 = (__m128i*)fvBuffers->d5;
        input_5 = fvBuffers->d5 + KK;
    case 5: 
        for (i = 0; i < K; i++) fvBuffers->d4[i] = I[i*N + 4];
        in_ptr4 = (__m128i*)fvBuffers->d4;
        input_4 = fvBuffers->d4 + KK;
    case 4: 
        for (i = 0; i < K; i++) fvBuffers->d3[i] = I[i*N + 3];
        in_ptr3 = (__m128i*)fvBuffers->d3;
        input_3 = fvBuffers->d3 + KK;
    case 3: 
        for (i = 0; i < K; i++) fvBuffers->d2[i] = I[i*N + 2];
        in_ptr2 = (__m128i*)fvBuffers->d2;
        input_2 = fvBuffers->d2 + KK;
    case 2: 
        for (i = 0; i < K; i++) fvBuffers->d1[i] = I[i*N + 1];
        in_ptr1 = (__m128i*)fvBuffers->d1;
        input_1 = fvBuffers->d1 + KK;
        for (i = 0; i < K; i++) fvBuffers->d0[i] = I[i*N];
        in_ptr0 = (__m128i*)fvBuffers->d0;
        input_0 = fvBuffers->d0 + KK;
    }
    ix_end = KK / SSE_16CAP;

    if (2 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
            }

            y += N;
        }

        return;
    }

    if (3 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
                y[2] += input_2[i] * *weight;
            }

            y += N;
        }

        return;
    }

    if (4 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);
            y[3] = *b + vec_sum(acc3);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
                y[2] += input_2[i] * *weight;
                y[3] += input_3[i] * *weight;
            }

            y += N;
        }

        return;
    }

    if (5 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);
            y[3] = *b + vec_sum(acc3);
            y[4] = *b + vec_sum(acc4);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
                y[2] += input_2[i] * *weight;
                y[3] += input_3[i] * *weight;
                y[4] += input_4[i] * *weight;
            }

            y += N;
        }

        return;
    }

    if (6 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);
            y[3] = *b + vec_sum(acc3);
            y[4] = *b + vec_sum(acc4);
            y[5] = *b + vec_sum(acc5);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
                y[2] += input_2[i] * *weight;
                y[3] += input_3[i] * *weight;
                y[4] += input_4[i] * *weight;
                y[5] += input_5[i] * *weight;
            }

            y += N;
        }

        return;
    }

    if (7 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);
                in6 = _mm_load_si128(in_ptr6 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);
                in6 = _mm_madd_epi16(in6, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
                acc6 = _mm_add_epi32(acc6, in6);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);
            y[3] = *b + vec_sum(acc3);
            y[4] = *b + vec_sum(acc4);
            y[5] = *b + vec_sum(acc5);
            y[6] = *b + vec_sum(acc6);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
                y[2] += input_2[i] * *weight;
                y[3] += input_3[i] * *weight;
                y[4] += input_4[i] * *weight;
                y[5] += input_5[i] * *weight;
                y[6] += input_6[i] * *weight;
            }

            y += N;
        }

        return;
    }

    if (8 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();
            acc7 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_load_si128(in_ptr0 + ix), w));
                acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_load_si128(in_ptr1 + ix), w));
                acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_load_si128(in_ptr2 + ix), w));
                acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_load_si128(in_ptr3 + ix), w));
                acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_load_si128(in_ptr4 + ix), w));
                acc5 = _mm_add_epi32(acc5, _mm_madd_epi16(_mm_load_si128(in_ptr5 + ix), w));
                acc6 = _mm_add_epi32(acc6, _mm_madd_epi16(_mm_load_si128(in_ptr6 + ix), w));
                acc7 = _mm_add_epi32(acc7, _mm_madd_epi16(_mm_load_si128(in_ptr7 + ix), w));
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);
            y[3] = *b + vec_sum(acc3);
            y[4] = *b + vec_sum(acc4);
            y[5] = *b + vec_sum(acc5);
            y[6] = *b + vec_sum(acc6);
            y[7] = *b + vec_sum(acc7);

            for (i = 0; i < KT; i++, weight++)
            {
                y[0] += input_0[i] * *weight;
                y[1] += input_1[i] * *weight;
                y[2] += input_2[i] * *weight;
                y[3] += input_3[i] * *weight;
                y[4] += input_4[i] * *weight;
                y[5] += input_5[i] * *weight;
                y[6] += input_6[i] * *weight;
                y[7] += input_7[i] * *weight;
            }
            y += 8;
        }

        return;
    }
}

void igemm16_subset_mb(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int16_t *W,
    const nn_bias_s *B,
    const uint32_t BG,
    int32_t *Y,
    const uint32_t *AL,
    const uint32_t L,
    uint32_t *nSat,
    KernelBuffers *bufs)
{}
