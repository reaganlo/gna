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

void igemm16(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int16_t *W,
    const nn_bias_s *B,
          int32_t *O,
          uint32_t* nSat,
          aligned_fv_bufs *fvBuffers)
{
    uint32_t i, j, k, kk, kpartial, nKpartial, niters;
    kpartial = (hw_buf_size[N - 1]) / N;
    nKpartial = K / kpartial;

    __m128i in[8], w;     // inputs & weight
    __m128i imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7, imm8, imm9, imm10;       // immediate
    __m128i acc[8]; // output accumulators
    __m128i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    uint32_t ix, ix_end;

    int16_t *input[8], *weight;// inputs & weight pointers
    nn_bias_s *bias;             // bias pointer
    int32_t *out;              // output pointer
    nn_bias_s *bias_end;         // outer loop pointer
    int64_t sum[8];            // 64-bit accumulator buffer

    uint32_t KT = K % SSE_16CAP; // K tail for manual processing
    uint32_t KK = K - KT; // trimmed K for AVX2 processing

    out = O;
    weight = const_cast<int16_t*>(W);
    bias = const_cast<nn_bias_s*>(B);
    bias_end = bias + M;

    if (1 == N)
    {
        *input = const_cast<int16_t*>(I)+KK;
        in_ptr0 = (__m128i*)I;
        for (; bias < bias_end; bias++)
        {
            *acc = _mm_setzero_si128();
            *sum = *bias;
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                *sum += vec_sum(*acc);
                *acc = _mm_setzero_si128();
                saturate(sum, nSat);

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    *in = _mm_load_si128(in_ptr0 + ix);
                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    *in = _mm_madd_epi16(*in, w);

                    // unpack to 64-bit
                    *in = _mm_add_epi64(_mm_cvtepi32_epi64(*in), _mm_cvtepi32_epi64(_mm_bsrli_si128(*in, 8)));

                    // accumulate
                    *acc = _mm_add_epi64(*acc, *in);
                }
            }

            *sum += vec_sum(*acc);

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
        in_ptr7 = (__m128i*)fvBuffers->d7;
        input[7] = fvBuffers->d7 + KK;
    case 7: 
        for (i = 0; i < K; i++) fvBuffers->d6[i] = I[i*N + 6];
        in_ptr6 = (__m128i*)fvBuffers->d6;
        input[6] = fvBuffers->d6 + KK;
    case 6: 
        for (i = 0; i < K; i++) fvBuffers->d5[i] = I[i*N + 5];
        in_ptr5 = (__m128i*)fvBuffers->d5;
        input[5] = fvBuffers->d5 + KK;
    case 5: 
        for (i = 0; i < K; i++) fvBuffers->d4[i] = I[i*N + 4];
        in_ptr4 = (__m128i*)fvBuffers->d4;
        input[4] = fvBuffers->d4 + KK;
    case 4: 
        for (i = 0; i < K; i++) fvBuffers->d3[i] = I[i*N + 3];
        in_ptr3 = (__m128i*)fvBuffers->d3;
        input[3] = fvBuffers->d3 + KK;
    case 3: 
        for (i = 0; i < K; i++) fvBuffers->d2[i] = I[i*N + 2];
        in_ptr2 = (__m128i*)fvBuffers->d2;
        input[2] = fvBuffers->d2 + KK;
    case 2: 
        for (i = 0; i < K; i++) fvBuffers->d1[i] = I[i*N + 1];
        in_ptr1 = (__m128i*)fvBuffers->d1;
        input[1] = fvBuffers->d1 + KK;
        for (i = 0; i < K; i++) fvBuffers->d0[i] = I[i*N];
        in_ptr0 = (__m128i*)fvBuffers->d0;
        input[0] = fvBuffers->d0 + KK;
    }

    if (2 == N)
    {
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);

                    // unpack to 64-bit
                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
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

    if (3 == N)
    {
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);
                    in[2] = _mm_load_si128(in_ptr2 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);
                    in[2] = _mm_madd_epi16(in[2], w);

                    // unpack to 64-bit
                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));
                    in[2] = _mm_add_epi64(_mm_cvtepi32_epi64(in[2]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[2], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
                    acc[2] = _mm_add_epi64(acc[2], in[2]);
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

    if (4 == N)
    {
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);
                    in[2] = _mm_load_si128(in_ptr2 + ix);
                    in[3] = _mm_load_si128(in_ptr3 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);
                    in[2] = _mm_madd_epi16(in[2], w);
                    in[3] = _mm_madd_epi16(in[3], w);

                    // unpack to 64-bit
                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));
                    in[2] = _mm_add_epi64(_mm_cvtepi32_epi64(in[2]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[2], 8)));
                    in[3] = _mm_add_epi64(_mm_cvtepi32_epi64(in[3]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[3], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
                    acc[2] = _mm_add_epi64(acc[2], in[2]);
                    acc[3] = _mm_add_epi64(acc[3], in[3]);

                    // load next vectors
                    in[0] = _mm_load_si128((__m128i*)input[0]);
                    in[1] = _mm_load_si128((__m128i*)input[1]);
                    in[2] = _mm_load_si128((__m128i*)input[2]);
                    in[3] = _mm_load_si128((__m128i*)input[3]);
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

    if (5 == N)
    {
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);
                    in[2] = _mm_load_si128(in_ptr2 + ix);
                    in[3] = _mm_load_si128(in_ptr3 + ix);
                    in[4] = _mm_load_si128(in_ptr4 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);
                    in[2] = _mm_madd_epi16(in[2], w);
                    in[3] = _mm_madd_epi16(in[3], w);
                    in[4] = _mm_madd_epi16(in[4], w);

                    // unpack to 64-bit
                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));
                    in[2] = _mm_add_epi64(_mm_cvtepi32_epi64(in[2]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[2], 8)));
                    in[3] = _mm_add_epi64(_mm_cvtepi32_epi64(in[3]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[3], 8)));
                    in[4] = _mm_add_epi64(_mm_cvtepi32_epi64(in[4]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[4], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
                    acc[2] = _mm_add_epi64(acc[2], in[2]);
                    acc[3] = _mm_add_epi64(acc[3], in[3]);
                    acc[4] = _mm_add_epi64(acc[4], in[4]);

                    // load next vectors
                    in[0] = _mm_load_si128((__m128i*)input[0]);
                    in[1] = _mm_load_si128((__m128i*)input[1]);
                    in[2] = _mm_load_si128((__m128i*)input[2]);
                    in[3] = _mm_load_si128((__m128i*)input[3]);
                    in[4] = _mm_load_si128((__m128i*)input[4]);
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
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);
                    in[2] = _mm_load_si128(in_ptr2 + ix);
                    in[3] = _mm_load_si128(in_ptr3 + ix);
                    in[4] = _mm_load_si128(in_ptr4 + ix);
                    in[5] = _mm_load_si128(in_ptr5 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);
                    in[2] = _mm_madd_epi16(in[2], w);
                    in[3] = _mm_madd_epi16(in[3], w);
                    in[4] = _mm_madd_epi16(in[4], w);
                    in[5] = _mm_madd_epi16(in[5], w);

                    // unpack to 64-bit
                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));
                    in[2] = _mm_add_epi64(_mm_cvtepi32_epi64(in[2]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[2], 8)));
                    in[3] = _mm_add_epi64(_mm_cvtepi32_epi64(in[3]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[3], 8)));
                    in[4] = _mm_add_epi64(_mm_cvtepi32_epi64(in[4]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[4], 8)));
                    in[5] = _mm_add_epi64(_mm_cvtepi32_epi64(in[5]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[5], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
                    acc[2] = _mm_add_epi64(acc[2], in[2]);
                    acc[3] = _mm_add_epi64(acc[3], in[3]);
                    acc[4] = _mm_add_epi64(acc[4], in[4]);
                    acc[5] = _mm_add_epi64(acc[5], in[5]);
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
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);
                    in[2] = _mm_load_si128(in_ptr2 + ix);
                    in[3] = _mm_load_si128(in_ptr3 + ix);
                    in[4] = _mm_load_si128(in_ptr4 + ix);
                    in[5] = _mm_load_si128(in_ptr5 + ix);
                    in[6] = _mm_load_si128(in_ptr6 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);
                    in[2] = _mm_madd_epi16(in[2], w);
                    in[3] = _mm_madd_epi16(in[3], w);
                    in[4] = _mm_madd_epi16(in[4], w);
                    in[5] = _mm_madd_epi16(in[5], w);
                    in[6] = _mm_madd_epi16(in[6], w);

                    // unpack to 64-bit
                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));
                    in[2] = _mm_add_epi64(_mm_cvtepi32_epi64(in[2]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[2], 8)));
                    in[3] = _mm_add_epi64(_mm_cvtepi32_epi64(in[3]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[3], 8)));
                    in[4] = _mm_add_epi64(_mm_cvtepi32_epi64(in[4]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[4], 8)));
                    in[5] = _mm_add_epi64(_mm_cvtepi32_epi64(in[5]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[5], 8)));
                    in[6] = _mm_add_epi64(_mm_cvtepi32_epi64(in[6]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[6], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
                    acc[2] = _mm_add_epi64(acc[2], in[2]);
                    acc[3] = _mm_add_epi64(acc[3], in[3]);
                    acc[4] = _mm_add_epi64(acc[4], in[4]);
                    acc[5] = _mm_add_epi64(acc[5], in[5]);
                    acc[6] = _mm_add_epi64(acc[6], in[6]);
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
        for (; bias < bias_end; bias++)
        {
            for (i = 0; i < N; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                sum[0] += vec_sum(acc[0]);
                sum[1] += vec_sum(acc[1]);
                sum[2] += vec_sum(acc[2]);
                sum[3] += vec_sum(acc[3]);
                sum[4] += vec_sum(acc[4]);
                sum[5] += vec_sum(acc[5]);
                sum[6] += vec_sum(acc[6]);
                sum[7] += vec_sum(acc[7]);

                for (i = 0; i < N; i++)
                {
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm_load_si128(in_ptr0 + ix);
                    in[1] = _mm_load_si128(in_ptr1 + ix);
                    in[2] = _mm_load_si128(in_ptr2 + ix);
                    in[3] = _mm_load_si128(in_ptr3 + ix);
                    in[4] = _mm_load_si128(in_ptr4 + ix);
                    in[5] = _mm_load_si128(in_ptr5 + ix);
                    in[6] = _mm_load_si128(in_ptr6 + ix);
                    in[7] = _mm_load_si128(in_ptr7 + ix);

                    w = _mm_lddqu_si128((__m128i*)weight);
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm_madd_epi16(in[0], w);
                    in[1] = _mm_madd_epi16(in[1], w);
                    in[2] = _mm_madd_epi16(in[2], w);
                    in[3] = _mm_madd_epi16(in[3], w);
                    in[4] = _mm_madd_epi16(in[4], w);
                    in[5] = _mm_madd_epi16(in[5], w);
                    in[6] = _mm_madd_epi16(in[6], w);
                    in[7] = _mm_madd_epi16(in[7], w);

                    in[0] = _mm_add_epi64(_mm_cvtepi32_epi64(in[0]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[0], 8)));
                    in[1] = _mm_add_epi64(_mm_cvtepi32_epi64(in[1]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[1], 8)));
                    in[2] = _mm_add_epi64(_mm_cvtepi32_epi64(in[2]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[2], 8)));
                    in[3] = _mm_add_epi64(_mm_cvtepi32_epi64(in[3]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[3], 8)));
                    in[4] = _mm_add_epi64(_mm_cvtepi32_epi64(in[4]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[4], 8)));
                    in[5] = _mm_add_epi64(_mm_cvtepi32_epi64(in[5]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[5], 8)));
                    in[6] = _mm_add_epi64(_mm_cvtepi32_epi64(in[6]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[6], 8)));
                    in[7] = _mm_add_epi64(_mm_cvtepi32_epi64(in[7]), _mm_cvtepi32_epi64(_mm_bsrli_si128(in[7], 8)));

                    // accumulate
                    acc[0] = _mm_add_epi64(acc[0], in[0]);
                    acc[1] = _mm_add_epi64(acc[1], in[1]);
                    acc[2] = _mm_add_epi64(acc[2], in[2]);
                    acc[3] = _mm_add_epi64(acc[3], in[3]);
                    acc[4] = _mm_add_epi64(acc[4], in[4]);
                    acc[5] = _mm_add_epi64(acc[5], in[5]);
                    acc[6] = _mm_add_epi64(acc[6], in[6]);
                    acc[7] = _mm_add_epi64(acc[7], in[7]);
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

void igemm16_mb(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int16_t *W,
    const nn_bias_s *B,
    const uint32_t BG,
    int32_t *Y,
    uint32_t *nSat,
    aligned_fv_bufs *bufs)
{}
