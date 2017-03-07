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

void igemm8(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int8_t *W,
    const nn_bias_c *B,
    int32_t *O,
    uint32_t *nSat,
    KernelBuffers *bufs)
{
    __m256i in0, in1, in2, in3, in4, in5, in6, in7;
    __m256i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    int64_t sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    uint32_t i, j, k, kk, kpartial, nKpartial, niters;
    uint32_t acc_iters, rem_iters;

    kpartial = (hw_buf_size[N - 1]) / N;
    nKpartial = K / kpartial;

    __m256i in[8], w;     // inputs & weight
    __m256i imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7, imm8, imm9, imm10;       // immediate
    __m256i acc[8]; // output accumulators
    __m256i zero = _mm256_setzero_si256(); // AVX2 ZERO

    int16_t *input[8];
    int8_t *weight;// inputs & weight pointers
    nn_bias_c *bias;             // bias pointer
    int32_t *out;              // output pointer
    nn_bias_c *bias_end;         // outer loop pointer
    int64_t sum[8];            // 64-bit accumulator buffer

    uint32_t KT = K % VEC_16CAP; // K tail for manual processing
    uint32_t KK = K - KT; // trimmed K for AVX2 processing

    out = O;
    weight = const_cast<int8_t*>(W);
    bias = const_cast<nn_bias_c*>(B);
    bias_end = bias + M;

    __m256i* in_ptr;
    uint32_t ix, ix_end;

    __m256i w0, w1;
    int16_t *input0;

    if (1 == N)
    {
        input0 = const_cast<int16_t*>(I)+KK;
        in_ptr = (__m256i*)I;
        for (; bias < bias_end; bias++)
        {
            ix = 0;
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            sum0 = bias->bias;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                acc0 = _mm256_add_epi32(acc0, acc1);
                sum0 += vec_sum32(acc0) * bias->multiplier;

                acc0 = _mm256_setzero_si256();
                acc1 = _mm256_setzero_si256();

                saturate(&sum0, nSat);
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12288
                // 12288 / 16 = 768
                // 12888 / 256 = 48
                // so, max number of loops is 3
                acc_iters = niters / (VEC_16CAP * 256);
                rem_iters = niters % (VEC_16CAP * 256);

                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm256_setzero_si256();
                    acc1 = _mm256_setzero_si256();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix += 2)
                    {
                        in0 = _mm256_load_si256(in_ptr + ix);
                        in1 = _mm256_load_si256(in_ptr + ix + 1);

                        w0 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                        w1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)(weight + VEC_16CAP)));

                        weight += 32;

                        // multiply and add - won't saturate
                        in0 = _mm256_madd_epi16(in0, w0);
                        in1 = _mm256_madd_epi16(in1, w1);

                        acc0 = _mm256_add_epi32(acc0, in0);
                        acc1 = _mm256_add_epi32(acc1, in1);

                        // load next vectors
                    }

                    acc0 = _mm256_add_epi32(acc0, acc1);
                    sum0 += vec_sum32(acc0) * bias->multiplier;
                }

                acc0 = _mm256_setzero_si256();
                acc1 = _mm256_setzero_si256();

                ix_end = ix + rem_iters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    // load next vectors
                    in0 = _mm256_load_si256(in_ptr + ix);
                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));

                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm256_madd_epi16(in0, w);
                    acc0 = _mm256_add_epi32(acc0, in0);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                acc0 = _mm256_setzero_si256();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input0[j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, out, nSat);

            out++;
        }
        return;
    }

    switch (N)
    {
    case 8: 
        for (i = 0; i < K; i++) bufs->d7[i] = I[i*N + 7];
        input[7] = bufs->d7 + KK;
        in_ptr7 = (__m256i*)bufs->d7;
    case 7: 
        for (i = 0; i < K; i++) bufs->d6[i] = I[i*N + 6];
        input[6] = bufs->d6 + KK;
        in_ptr6 = (__m256i*)bufs->d6;
    case 6: 
        for (i = 0; i < K; i++) bufs->d5[i] = I[i*N + 5];
        input[5] = bufs->d5 + KK;
        in_ptr5 = (__m256i*)bufs->d5;
    case 5: 
        for (i = 0; i < K; i++) bufs->d4[i] = I[i*N + 4];
        input[4] = bufs->d4 + KK;
        in_ptr4 = (__m256i*)bufs->d4;
    case 4: 
        for (i = 0; i < K; i++) bufs->d3[i] = I[i*N + 3];
        input[3] = bufs->d3 + KK;
        in_ptr3 = (__m256i*)bufs->d3;
    case 3: 
        for (i = 0; i < K; i++) bufs->d2[i] = I[i*N + 2];
        input[2] = bufs->d2 + KK;
        in_ptr2 = (__m256i*)bufs->d2;
    case 2: 
        for (i = 0; i < K; i++) bufs->d1[i] = I[i*N + 1];
        input[1] = bufs->d1 + KK;
        in_ptr1 = (__m256i*)bufs->d1;
        for (i = 0; i < K; i++) bufs->d0[i] = I[i*N];
        input[0] = bufs->d0 + KK;
        in_ptr0 = (__m256i*)bufs->d0;
    }

    if (2 == N)
    {
        for (; bias < bias_end; bias++)
        {
            ix = 0;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 6144
                // 6144 / 16 = 384
                // 6144 / 256 = 24
                // so, max number of loops is 1

                acc_iters = niters / (VEC_16CAP * 256);
                rem_iters = niters % (VEC_16CAP * 256);

                if (acc_iters == 1)
                {
                    acc[0] = _mm256_setzero_si256();
                    acc[1] = _mm256_setzero_si256();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                        weight += VEC_16CAP;

                        in[0] = _mm256_load_si256(in_ptr0 + ix);
                        in[1] = _mm256_load_si256(in_ptr1 + ix);

                        // multiply and add - won't saturate
                        in[0] = _mm256_madd_epi16(in[0], w);
                        in[1] = _mm256_madd_epi16(in[1], w);

                        acc[0] = _mm256_add_epi32(acc[0], in[0]);
                        acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    }

                    sum[0] += vec_sum32(acc[0]) * bias->multiplier;
                    sum[1] += vec_sum32(acc[1]) * bias->multiplier;
                }

                acc[0] = _mm256_setzero_si256();
                acc[1] = _mm256_setzero_si256();

                ix_end = ix + rem_iters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // load next vectors
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                }

                sum[0] += vec_sum32(acc[0]) * bias->multiplier;
                sum[1] += vec_sum32(acc[1]) * bias->multiplier;

                acc[0] = _mm256_setzero_si256();
                acc[1] = _mm256_setzero_si256();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum[0] += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum[1] += (int32_t)(input[1][j] * *weight * bias->multiplier);
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
            ix = 0;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12096 / 3 = 4032
                // 4032 / 16 = 252
                // accumulator will not saturate
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    // load next vectors
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                }

                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += input[i][j] * *weight * bias->multiplier;
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
            ix = 0;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                // kpartial = 12288 / 4 = 3072
                // 3072 / 16 = 192
                // accumulator will not saturate

                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    // load next vectors
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                }

                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight * bias->multiplier);
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
            ix = 0;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12000 / 5 = 2400
                // 2400 / 16 = 150
                // accumulator will not saturate
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                    acc[4] = _mm256_add_epi32(acc[4], in[4]);
                }

                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight * bias->multiplier);
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
            ix = 0;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 12288 / 6 = 2048
                // 2048 / 16 = 128
                // accumulator will not saturate
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in[0] = _mm256_load_si256(in_ptr0 + ix);
                    in[1] = _mm256_load_si256(in_ptr1 + ix);
                    in[2] = _mm256_load_si256(in_ptr2 + ix);
                    in[3] = _mm256_load_si256(in_ptr3 + ix);
                    in[4] = _mm256_load_si256(in_ptr4 + ix);
                    in[5] = _mm256_load_si256(in_ptr5 + ix);
                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                    acc[4] = _mm256_add_epi32(acc[4], in[4]);
                    acc[5] = _mm256_add_epi32(acc[5], in[5]);
                }

                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight * bias->multiplier);
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
            ix = 0;

            for (i = 0; i < N; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = bias->bias;
            }

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], nSat);
                }
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                // kpartial = 12288 / 7 = 1755
                // kpartial / 16 = 109
                // accumulator will not saturate

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

                    w = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                    weight += VEC_16CAP;

                    // multiply and add - won't saturate
                    in[0] = _mm256_madd_epi16(in[0], w);
                    in[1] = _mm256_madd_epi16(in[1], w);
                    in[2] = _mm256_madd_epi16(in[2], w);
                    in[3] = _mm256_madd_epi16(in[3], w);
                    in[4] = _mm256_madd_epi16(in[4], w);
                    in[5] = _mm256_madd_epi16(in[5], w);
                    in[6] = _mm256_madd_epi16(in[6], w);

                    acc[0] = _mm256_add_epi32(acc[0], in[0]);
                    acc[1] = _mm256_add_epi32(acc[1], in[1]);
                    acc[2] = _mm256_add_epi32(acc[2], in[2]);
                    acc[3] = _mm256_add_epi32(acc[3], in[3]);
                    acc[4] = _mm256_add_epi32(acc[4], in[4]);
                    acc[5] = _mm256_add_epi32(acc[5], in[5]);
                    acc[6] = _mm256_add_epi32(acc[6], in[6]);
                }

                for (i = 0; i < N; i++)
                {
                    sum[i] += vec_sum32(acc[i]) * bias->multiplier;
                    acc[i] = _mm256_setzero_si256();
                }
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < N; i++)
                {
                    sum[i] += (int32_t)(input[i][j] * *weight * bias->multiplier);
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
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;
            sum5 = bias->bias;
            sum6 = bias->bias;
            sum7 = bias->bias;

            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();
            acc7 = _mm256_setzero_si256();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);
                saturate(&sum2, nSat);
                saturate(&sum3, nSat);
                saturate(&sum4, nSat);
                saturate(&sum5, nSat);
                saturate(&sum6, nSat);
                saturate(&sum7, nSat);

                // kpartial = 12288 / 8 = 1536
                // 1536 / 16 = 96
                // accumulator will not saturate
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                ix_end = ix + niters / VEC_16CAP;
                for (; ix < ix_end; ix++)
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

                    // multiply and add - won't saturate
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

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
                sum4 += vec_sum32(acc4) * bias->multiplier;
                sum5 += vec_sum32(acc5) * bias->multiplier;
                sum6 += vec_sum32(acc6) * bias->multiplier;
                sum7 += vec_sum32(acc7) * bias->multiplier;

                acc0 = _mm256_setzero_si256();
                acc1 = _mm256_setzero_si256();
                acc2 = _mm256_setzero_si256();
                acc3 = _mm256_setzero_si256();
                acc4 = _mm256_setzero_si256();
                acc5 = _mm256_setzero_si256();
                acc6 = _mm256_setzero_si256();
                acc7 = _mm256_setzero_si256();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum1 += (int32_t)(input[1][j] * *weight * bias->multiplier);
                sum2 += (int32_t)(input[2][j] * *weight * bias->multiplier);
                sum3 += (int32_t)(input[3][j] * *weight * bias->multiplier);
                sum4 += (int32_t)(input[4][j] * *weight * bias->multiplier);
                sum5 += (int32_t)(input[5][j] * *weight * bias->multiplier);
                sum6 += (int32_t)(input[6][j] * *weight * bias->multiplier);
                sum7 += (int32_t)(input[7][j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);
            saturate_store_out(&sum2, &out[2], nSat);
            saturate_store_out(&sum3, &out[3], nSat);
            saturate_store_out(&sum4, &out[4], nSat);
            saturate_store_out(&sum5, &out[5], nSat);
            saturate_store_out(&sum6, &out[6], nSat);
            saturate_store_out(&sum7, &out[7], nSat);

            out += N;
        }
    }
}

void
igemm8_mb(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int8_t*     W,
    const   nn_bias_s*  B,
    const   uint32_t    BG,
    const   nn_bias_c*  CB,
    int32_t*    O,
    uint32_t*   nSat,
    KernelBuffers*    fvBuffers)
{}

