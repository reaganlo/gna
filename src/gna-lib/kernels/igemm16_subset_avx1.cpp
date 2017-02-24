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
#include "string.h"

void igemm16_subset(
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
          aligned_fv_bufs *bufs,
          const int biasShift)
{
    uint32_t i, j, k, l, ix, ix_end;

    int32_t *y = Y;
    int16_t *weight = const_cast<int16_t*>(W);

    __m256i v0, v1, v2, v3;
    __m128i s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, w, w0, w1;
    __m256i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    int16_t *input_0, *input_1, *input_2, *input_3, *input_4, *input_5, *input_6, *input_7;

    nn_bias_s *b, *be = const_cast<nn_bias_s*>(B)+M;

    uint32_t KT = K % VEC_16CAP;
    uint32_t KK = K - KT;
    ix_end = KK / VEC_16CAP;

    __m128i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m128i in0, in1, in2, in3, in4, in5, in6, in7;

    if (1 == N)
    {
        in_ptr0 = (__m256i*)I;
        input_0 = const_cast<int16_t*>(I)+KK;
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));

                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc0 = _mm_add_epi32(acc0, in1);
            }

            *y = *b + vec_sum(acc0);
            for (j = 0; j < KT; j++, weight++)
            {
                *y += input_0[j] * *weight;
            }
            y++;
        }

        return;
    }

    in_ptr0 = (__m256i*)bufs->d0;
    in_ptr1 = (__m256i*)bufs->d1;
    in_ptr2 = (__m256i*)bufs->d2;
    in_ptr3 = (__m256i*)bufs->d3;

    input_0 = bufs->d0 + KK;
    input_1 = bufs->d1 + KK;
    input_2 = bufs->d2 + KK;
    input_3 = bufs->d3 + KK;

    switch (N)
    {
    case 4: for (i = 0; i < K; i++) bufs->d3[i] = I[i*N + 3];
    case 3: for (i = 0; i < K; i++) bufs->d2[i] = I[i*N + 2];
    case 2: for (i = 0; i < K; i++) bufs->d1[i] = I[i*N + 1];
        for (i = 0; i < K; i++) bufs->d0[i] = I[i*N];
    }

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
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);

                in2 = _mm256_extractf128_si256(v0, 1);
                in3 = _mm256_extractf128_si256(v1, 1);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);

                in2 = _mm_madd_epi16(in2, w1);
                in3 = _mm_madd_epi16(in3, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);

                acc0 = _mm_add_epi32(acc0, in2);
                acc1 = _mm_add_epi32(acc1, in3);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_1[j] * *weight;
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
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));
                weight += VEC_16CAP;

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);
                in2 = _mm256_castsi256_si128(v2);

                in3 = _mm256_extractf128_si256(v0, 1);
                in4 = _mm256_extractf128_si256(v1, 1);
                in5 = _mm256_extractf128_si256(v2, 1);

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);
                in2 = _mm_madd_epi16(in2, w0);

                in3 = _mm_madd_epi16(in3, w1);
                in4 = _mm_madd_epi16(in4, w1);
                in5 = _mm_madd_epi16(in5, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);

                acc0 = _mm_add_epi32(acc0, in3);
                acc1 = _mm_add_epi32(acc1, in4);
                acc2 = _mm_add_epi32(acc2, in5);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_1[j] * *weight;
                y[2] += input_2[j] * *weight;
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
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);
                v3 = _mm256_load_si256(in_ptr3 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_castsi256_si128(v1);
                in2 = _mm256_castsi256_si128(v2);
                in3 = _mm256_castsi256_si128(v3);

                in4 = _mm256_extractf128_si256(v0, 1);
                in5 = _mm256_extractf128_si256(v1, 1);
                in6 = _mm256_extractf128_si256(v2, 1);
                in7 = _mm256_extractf128_si256(v3, 1);

                w0 = _mm_lddqu_si128((__m128i*)weight);
                w1 = _mm_lddqu_si128((__m128i*)(weight + SSE_16CAP));

                weight += VEC_16CAP;

                in0 = _mm_madd_epi16(in0, w0);
                in1 = _mm_madd_epi16(in1, w0);
                in2 = _mm_madd_epi16(in2, w0);
                in3 = _mm_madd_epi16(in3, w0);

                in4 = _mm_madd_epi16(in4, w1);
                in5 = _mm_madd_epi16(in5, w1);
                in6 = _mm_madd_epi16(in6, w1);
                in7 = _mm_madd_epi16(in7, w1);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);

                acc0 = _mm_add_epi32(acc0, in4);
                acc1 = _mm_add_epi32(acc1, in5);
                acc2 = _mm_add_epi32(acc2, in6);
                acc3 = _mm_add_epi32(acc3, in7);
            }

            y[0] = *b + vec_sum(acc0);
            y[1] = *b + vec_sum(acc1);
            y[2] = *b + vec_sum(acc2);
            y[3] = *b + vec_sum(acc3);

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_1[j] * *weight;
                y[2] += input_2[j] * *weight;
                y[3] += input_3[j] * *weight;
            }
            y += 4;
        }

        return;
    }

    KT = K % SSE_16CAP;
    KK = K - KT;
    ix_end = 2 * KK / VEC_16CAP;

    input_0 = bufs->d0 + 2 * KK;
    input_1 = bufs->d2 + 2 * KK;
    input_2 = bufs->d4 + 2 * KK;
    input_3 = bufs->d6 + 2 * KK;

    in_ptr0 = (__m256i*)bufs->d0;
    in_ptr1 = (__m256i*)bufs->d2;
    in_ptr2 = (__m256i*)bufs->d4;
    in_ptr3 = (__m256i*)bufs->d6;

    if (5 == N)
    {
        for (j = 0; j < 2 * K;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N];
                bufs->d2[j] = I[k*N + 2];
            }
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N + 1];
                bufs->d2[j] = I[k*N + 3];
            }
        }
        for (i = 0; i < K; i++) bufs->d4[i] = I[i*N + 4];

        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            input_2 = bufs->d4;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm_load_si128((__m128i*)input_2);
                w = _mm_lddqu_si128((__m128i*)weight);
                input_2 += SSE_16CAP;
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

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_0[j + KT] * *weight;
                y[2] += input_1[j] * *weight;
                y[3] += input_1[j + KT] * *weight;
                y[4] += input_2[j] * *weight;
            }

            y += N;
        }

        return;
    }

    if (6 == N)
    {
        for (j = 0; j < 2 * K;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N];
                bufs->d2[j] = I[k*N + 2];
                bufs->d4[j] = I[k*N + 4];
            }
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N + 1];
                bufs->d2[j] = I[k*N + 3];
                bufs->d4[j] = I[k*N + 5];
            }
        }

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
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
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

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_0[j + KT] * *weight;
                y[2] += input_1[j] * *weight;
                y[3] += input_1[j + KT] * *weight;
                y[4] += input_2[j] * *weight;
                y[5] += input_2[j + KT] * *weight;
            }

            y += N;
        }

        return;
    }

    if (7 == N)
    {
        for (j = 0; j < 2 * K;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N];
                bufs->d2[j] = I[k*N + 2];
                bufs->d4[j] = I[k*N + 4];
            }
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N + 1];
                bufs->d2[j] = I[k*N + 3];
                bufs->d4[j] = I[k*N + 5];
            }
        }

        for (i = 0; i < K; i++) bufs->d6[i] = I[i*N + 6];

        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            b = const_cast<nn_bias_s*>(B)+i;

            input_3 = bufs->d6;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                in6 = _mm_load_si128((__m128i*)input_3);
                w = _mm_lddqu_si128((__m128i*)weight);
                input_3 += SSE_16CAP;
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

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_0[j + KT] * *weight;
                y[2] += input_1[j] * *weight;
                y[3] += input_1[j + KT] * *weight;
                y[4] += input_2[j] * *weight;
                y[5] += input_2[j + KT] * *weight;
                y[6] += input_3[j] * *weight;
            }

            y += N;
        }
    }

    if (8 == N)
    {
        for (j = 0; j < 2 * K;)
        {
            i = j / 2;
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N];
                bufs->d2[j] = I[k*N + 2];
                bufs->d4[j] = I[k*N + 4];
                bufs->d6[j] = I[k*N + 6];
            }
            for (k = i; k < i + 8 && k < K; k++, j++)
            {
                bufs->d0[j] = I[k*N + 1];
                bufs->d2[j] = I[k*N + 3];
                bufs->d4[j] = I[k*N + 5];
                bufs->d6[j] = I[k*N + 7];
            }
        }

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
                v0 = _mm256_load_si256(in_ptr0 + ix);
                v1 = _mm256_load_si256(in_ptr1 + ix);
                v2 = _mm256_load_si256(in_ptr2 + ix);
                v3 = _mm256_load_si256(in_ptr3 + ix);

                in0 = _mm256_castsi256_si128(v0);
                in1 = _mm256_extractf128_si256(v0, 1);
                in2 = _mm256_castsi256_si128(v1);
                in3 = _mm256_extractf128_si256(v1, 1);
                in4 = _mm256_castsi256_si128(v2);
                in5 = _mm256_extractf128_si256(v2, 1);
                in6 = _mm256_castsi256_si128(v3);
                in7 = _mm256_extractf128_si256(v3, 1);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);
                in4 = _mm_madd_epi16(in4, w);
                in5 = _mm_madd_epi16(in5, w);
                in6 = _mm_madd_epi16(in6, w);
                in7 = _mm_madd_epi16(in7, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
                acc4 = _mm_add_epi32(acc4, in4);
                acc5 = _mm_add_epi32(acc5, in5);
                acc6 = _mm_add_epi32(acc6, in6);
                acc7 = _mm_add_epi32(acc7, in7);
            }

            s1 = _mm_set1_epi32(*b);

            s2 = _mm_hadd_epi32(acc0, acc1);
            s3 = _mm_hadd_epi32(acc2, acc3);
            s4 = _mm_hadd_epi32(s2, s3);
            s5 = _mm_add_epi32(s4, s1);

            s2 = _mm_hadd_epi32(acc4, acc5);
            s3 = _mm_hadd_epi32(acc6, acc7);
            s4 = _mm_hadd_epi32(s2, s3);
            s6 = _mm_add_epi32(s4, s1);

            v0 = _mm256_set_m128i(s6, s5);
            _mm256_stream_si256((__m256i*)y, v0);

            for (j = 0; j < KT; j++, weight++)
            {
                y[0] += input_0[j] * *weight;
                y[1] += input_0[j + KT] * *weight;
                y[2] += input_1[j] * *weight;
                y[3] += input_1[j + KT] * *weight;
                y[4] += input_2[j] * *weight;
                y[5] += input_2[j + KT] * *weight;
                y[6] += input_3[j] * *weight;
                y[7] += input_3[j + KT] * *weight;
            }

            y += N;
        }

        return;
    }
}
