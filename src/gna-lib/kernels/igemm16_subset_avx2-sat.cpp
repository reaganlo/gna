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

void igemm16_subset(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int16_t *W,
    const nn_bias_s *B,
          int32_t *O,
    const uint32_t *AL,
    const uint32_t L,
          uint32_t *nSat,
          aligned_fv_bufs *fvBuffers)
{
    uint32_t i, ix, ix_end, j, k, l, kk, kpartial, nKpartial, niters;
    kpartial = (hw_buf_size[N - 1]) / N;
    nKpartial = K / kpartial;

    __m256i in[8], w;     // inputs & weight
    __m256i imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7, imm8, imm9, imm10;       // immediate
    __m256i acc[8]; // output accumulators
    __m256i zero = _mm256_setzero_si256(); // AVX2 ZERO
    __m256i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    int16_t *input_0, *input_1, *input_2, *input_3, *input_4, *input_5, *input_6, *input_7;

    int16_t *input[8], *weight;// inputs & weight pointers
    nn_bias_s *bias;             // bias pointer
    int32_t *out;              // output pointer
    nn_bias_s *bias_end;         // outer loop pointer
    int64_t sum[8];            // 64-bit accumulator buffer

    uint32_t KT = K % VEC_16CAP; // K tail for manual processing
    uint32_t KK = K - KT; // trimmed K for AVX2 processing

    out = O;
    weight = const_cast<int16_t*>(W);
    bias = const_cast<nn_bias_s*>(B);
    bias_end = bias + M;

    if (1 == N)
    {
        in_ptr0 = (__m256i*)I;
        *input = const_cast<int16_t*>(I)+KK;
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            *acc = _mm256_setzero_si256();
            *sum = *bias;
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(sum, nSat);

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    *in = _mm256_load_si256(in_ptr0 + ix);
                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    *in = _mm256_madd_epi16(*in, w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(*in));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(*in, 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    *acc = _mm256_add_epi64(*acc, imm2);
                }

                *sum += vec_sum(acc[0]);
                *acc = _mm256_setzero_si256();
            }

            *sum += vec_sum(acc[0]);

            for (j = 0; j < KT; j++, weight++)
            {
                *sum += (int32_t)((*input)[j] * *weight);
            }

            saturate_store_out(sum, out, nSat);

            out++;
        }
        return;
    }

    switch (N)
    {
    case 8: 
        for (i = 0; i < K; i++) fvBuffers->d7[i] = I[i*N + 7];
        in_ptr7 = (__m256i*)fvBuffers->d7;
        input[7] = fvBuffers->d7 + KK;
    case 7: 
        for (i = 0; i < K; i++) fvBuffers->d6[i] = I[i*N + 6];
        in_ptr6 = (__m256i*)fvBuffers->d6;
        input[6] = fvBuffers->d6 + KK;
    case 6: 
        for (i = 0; i < K; i++) fvBuffers->d5[i] = I[i*N + 5];
        in_ptr5 = (__m256i*)fvBuffers->d5;
        input[5] = fvBuffers->d5 + KK;
    case 5: 
        for (i = 0; i < K; i++) fvBuffers->d4[i] = I[i*N + 4];
        in_ptr4 = (__m256i*)fvBuffers->d4;
        input[4] = fvBuffers->d4 + KK;
    case 4: 
        for (i = 0; i < K; i++) fvBuffers->d3[i] = I[i*N + 3];
        in_ptr3 = (__m256i*)fvBuffers->d3;
        input[3] = fvBuffers->d3 + KK;
    case 3: 
        for (i = 0; i < K; i++) fvBuffers->d2[i] = I[i*N + 2];
        in_ptr2 = (__m256i*)fvBuffers->d2;
        input[2] = fvBuffers->d2 + KK;
    case 2: 
        for (i = 0; i < K; i++) fvBuffers->d1[i] = I[i*N + 1];
        in_ptr1 = (__m256i*)fvBuffers->d1;
        input[1] = fvBuffers->d1 + KK;
        
        for (i = 0; i < K; i++) fvBuffers->d0[i] = I[i*N];
        in_ptr0 = (__m256i*)fvBuffers->d0;
        input[0] = fvBuffers->d0 + KK;
    }

    if (2 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();

            sum[0] = *bias;
            sum[1] = *bias;

            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);

                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += (int32_t)(input[0][j] * *weight);
                sum[1] += (int32_t)(input[1][j] * *weight);
            }

            saturate_store_out(&sum[0], &out[0], nSat);
            saturate_store_out(&sum[1], &out[1], nSat);

            out += N;
        }
    }

    if (3 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();

            sum[0] = *bias;
            sum[1] = *bias;
            sum[2] = *bias;
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);
            sum[2] += vec_sum(acc[2]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += (int32_t)(input[0][j] * *weight);
                sum[1] += (int32_t)(input[1][j] * *weight);
                sum[2] += (int32_t)(input[2][j] * *weight);
            }

            saturate_store_out(&sum[0], &out[0], nSat);
            saturate_store_out(&sum[1], &out[1], nSat);
            saturate_store_out(&sum[2], &out[2], nSat);

            out += N;
        }
    }

    if (4 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                }
            }

            sum[0] += vec_sum(acc[0]);
            sum[1] += vec_sum(acc[1]);
            sum[2] += vec_sum(acc[2]);
            sum[3] += vec_sum(acc[3]);

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += (int32_t)(input[0][j] * *weight);
                sum[1] += (int32_t)(input[1][j] * *weight);
                sum[2] += (int32_t)(input[2][j] * *weight);
                sum[3] += (int32_t)(input[3][j] * *weight);
            }

            saturate_store_out(&sum[0], &out[0], nSat);
            saturate_store_out(&sum[1], &out[1], nSat);
            saturate_store_out(&sum[2], &out[2], nSat);
            saturate_store_out(&sum[3], &out[3], nSat);

            out += N;
        }
    }

    if (5 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                }
            }

            for (i = 0; i < N; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight);
                }
            }

            for (i = 0; i < N; i++)
            {
                saturate_store_out(&sum[i], &out[i], nSat);
            }

            out += N;
        }
    }

    if (6 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                }
            }

            for (i = 0; i < N; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight);
                }
            }

            for (i = 0; i < N; i++)
            {
                saturate_store_out(&sum[i], &out[i], nSat);
            }

            out += N;
        }
    }

    if (7 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    in[6] = _mm256_load_si256(in_ptr6 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[6]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[6], 1));
                    imm8 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                    acc[6] = _mm256_add_epi64(acc[6], imm8);
                }
            }

            for (i = 0; i < N; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight);
                }
            }

            for (i = 0; i < N; i++)
            {
                saturate_store_out(&sum[i], &out[i], nSat);
            }

            out += N;
        }
    }

    if (8 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int16_t*>(W)+i*K;
            bias = const_cast<nn_bias_s*>(B)+i;

            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();
            acc[3] = _mm256_setzero_si256();
            acc[4] = _mm256_setzero_si256();
            acc[5] = _mm256_setzero_si256();
            acc[6] = _mm256_setzero_si256();
            acc[7] = _mm256_setzero_si256();

            for (i = 0; i < N; i++)
            {
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    in[6] = _mm256_load_si256(in_ptr6 + ix);
                    in[7] = _mm256_load_si256(in_ptr7 + ix);

                    w = _mm256_lddqu_si256((__m256i*)weight);
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);
                    in[7] = _mm256_madd_epi16(in[7], w);

                    // unpack to 64-bit
                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[0]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[0], 1));
                    imm2 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[1]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[1], 1));
                    imm3 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[2]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[2], 1));
                    imm4 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[3]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[3], 1));
                    imm5 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[4]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[4], 1));
                    imm6 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[5]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[5], 1));
                    imm7 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[6]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[6], 1));
                    imm8 = _mm256_add_epi64(imm0, imm1);

                    imm0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(in[7]));
                    imm1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(in[7], 1));
                    imm9 = _mm256_add_epi64(imm0, imm1);

                    // accumulate
                    acc[0] = _mm256_add_epi64(acc[0], imm2);
                    acc[1] = _mm256_add_epi64(acc[1], imm3);
                    acc[2] = _mm256_add_epi64(acc[2], imm4);
                    acc[3] = _mm256_add_epi64(acc[3], imm5);
                    acc[4] = _mm256_add_epi64(acc[4], imm6);
                    acc[5] = _mm256_add_epi64(acc[5], imm7);
                    acc[6] = _mm256_add_epi64(acc[6], imm8);
                    acc[7] = _mm256_add_epi64(acc[7], imm9);
                }
            }

            for (i = 0; i < N; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight);
                }
            }

            for (i = 0; i < N; i++)
            {
                saturate_store_out(&sum[i], out++, nSat);
            }
        }
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
    aligned_fv_bufs *bufs)
{}
