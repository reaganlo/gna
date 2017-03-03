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

void igemm8_subset(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int8_t *W,
    const nn_bias_c *B,
          int32_t *O,
    const uint32_t *AL,
    const uint32_t L,
          uint32_t *nSat,
          aligned_fv_bufs *bufs)
{
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t i, j, k, l, kk, kpartial, nKpartial, niters;
    kpartial = (hw_buf_size[N - 1]) / N;
    nKpartial = K / kpartial;

    __m128i in[8], w, w0, w1;     // inputs & weight
    __m128i imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7, imm8, imm9, imm10;       // immediate
    __m128i acc[8]; // output accumulators
    __m128i zero = _mm_setzero_si128(); // AVX2 ZERO

    int16_t *input[8];
    int8_t *weight;             // inputs & weight pointers
    nn_bias_c *bias;             // bias pointer
    int32_t *out;              // output pointer
    nn_bias_c *bias_end;         // outer loop pointer
    int64_t sum[8];            // 64-bit accumulator buffer

    uint32_t KT = K % SSE_16CAP; // K tail for manual processing
    uint32_t KK = K - KT; // trimmed K for AVX2 processing

    out = O;
    weight = const_cast<int8_t*>(W);
    bias = const_cast<nn_bias_c*>(B);
    bias_end = bias + M;

    __m128i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    __m128i in0, in1, in2, in3, in4, in5, in6, in7;
    __m128i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    int64_t sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    uint32_t ix, ix_end;

    if (1 == N)
    {
        in_ptr0 = (__m128i*)I;
        input[0] = const_cast<int16_t*>(I)+KK;
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;

            ix = 0;
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            sum0 = bias->bias;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                acc0 = _mm_add_epi32(acc0, acc1);
                sum0 += vec_sum32(acc0) * bias->multiplier;
                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                saturate_store_out(&sum0, out, nSat);
                sum0 = *out;

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                acc_iters = niters / (256 * SSE_16CAP);
                rem_iters = niters % (256 * SSE_16CAP);

                // kpartial is 12288
                // 12288 / 256 = 48
                // max iters = 48 / SSE_16CAP = 6
                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix += 2)
                    {
                        in0 = _mm_load_si128(in_ptr0 + ix);
                        in1 = _mm_load_si128(in_ptr0 + ix + 1);

                        w0 = _mm_cvtepi8_epi16(_mm_loadu_si64(weight));
                        w1 = _mm_cvtepi8_epi16(_mm_loadu_si64(weight + SSE_16CAP));
                        weight += 2 * SSE_16CAP;

                        // multiply and add - won't saturate
                        in0 = _mm_madd_epi16(in0, w0);
                        in1 = _mm_madd_epi16(in1, w1);
                        acc0 = _mm_add_epi32(acc0, in0);
                        acc1 = _mm_add_epi32(acc1, in1);
                    }

                    acc0 = _mm_add_epi32(acc0, acc1);
                    sum0 += vec_sum32(acc0) * bias->multiplier;
                }

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();

                ix_end = ix + rem_iters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);

                    w = _mm_cvtepi8_epi16(_mm_loadu_si64(weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    acc0 = _mm_add_epi32(acc0, in0);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                acc0 = _mm_setzero_si128();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)((*input)[j] * *weight++ * bias->multiplier);
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
        in_ptr7 = (__m128i*)bufs->d7;
    case 7: 
        for (i = 0; i < K; i++) bufs->d6[i] = I[i*N + 6];
        input[6] = bufs->d6 + KK;
        in_ptr6 = (__m128i*)bufs->d6;
    case 6: 
        for (i = 0; i < K; i++) bufs->d5[i] = I[i*N + 5];
        input[5] = bufs->d5 + KK;
        in_ptr5 = (__m128i*)bufs->d5;
    case 5: 
        for (i = 0; i < K; i++) bufs->d4[i] = I[i*N + 4];
        input[4] = bufs->d4 + KK;
        in_ptr4 = (__m128i*)bufs->d4;
    case 4: 
        for (i = 0; i < K; i++) bufs->d3[i] = I[i*N + 3];
        input[3] = bufs->d3 + KK;
        in_ptr3 = (__m128i*)bufs->d3;
    case 3: 
        for (i = 0; i < K; i++) bufs->d2[i] = I[i*N + 2];
        input[2] = bufs->d2 + KK;
        in_ptr2 = (__m128i*)bufs->d2;
    case 2: 
        for (i = 0; i < K; i++) bufs->d1[i] = I[i*N + 1];
        input[1] = bufs->d1 + KK;
        in_ptr1 = (__m128i*)bufs->d1;
        
        for (i = 0; i < K; i++) bufs->d0[i] = I[i*N];
        input[0] = bufs->d0 + KK;
        in_ptr0 = (__m128i*)bufs->d0;
    }

    if (2 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);

                // kpartial = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                acc_iters = niters / (256 * SSE_16CAP);
                rem_iters = niters % (256 * SSE_16CAP);

                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr0 + ix);
                        in1 = _mm_load_si128(in_ptr1 + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                        weight += SSE_16CAP;

                        // multiply and add - won't saturate
                        in0 = _mm_madd_epi16(in0, w);
                        in1 = _mm_madd_epi16(in1, w);

                        acc0 = _mm_add_epi32(acc0, in0);
                        acc1 = _mm_add_epi32(acc1, in1);
                    }

                    sum0 += vec_sum32(acc0) * bias->multiplier;
                    sum1 += vec_sum32(acc1) * bias->multiplier;
                }

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();

                ix_end = ix + rem_iters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    in1 = _mm_madd_epi16(in1, w);

                    acc0 = _mm_add_epi32(acc0, in0);
                    acc1 = _mm_add_epi32(acc1, in1);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum1 += (int32_t)(input[1][j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);

            out += N;
        }
    }

    if (3 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);
                saturate(&sum2, nSat);

                // kpartial = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                acc_iters = niters / (256 * SSE_16CAP);
                rem_iters = niters % (256 * SSE_16CAP);

                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();
                    acc2 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr0 + ix);
                        in1 = _mm_load_si128(in_ptr1 + ix);
                        in2 = _mm_load_si128(in_ptr2 + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                        weight += SSE_16CAP;

                        // multiply and add - won't saturate
                        in0 = _mm_madd_epi16(in0, w);
                        in1 = _mm_madd_epi16(in1, w);
                        in2 = _mm_madd_epi16(in2, w);

                        acc0 = _mm_add_epi32(acc0, in0);
                        acc1 = _mm_add_epi32(acc1, in1);
                        acc2 = _mm_add_epi32(acc2, in2);
                    }

                    sum0 += vec_sum32(acc0) * bias->multiplier;
                    sum1 += vec_sum32(acc1) * bias->multiplier;
                    sum2 += vec_sum32(acc2) * bias->multiplier;
                }

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();

                ix_end = ix + rem_iters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    in2 = _mm_load_si128(in_ptr2 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    in1 = _mm_madd_epi16(in1, w);
                    in2 = _mm_madd_epi16(in2, w);

                    acc0 = _mm_add_epi32(acc0, in0);
                    acc1 = _mm_add_epi32(acc1, in1);
                    acc2 = _mm_add_epi32(acc2, in2);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum1 += (int32_t)(input[1][j] * *weight * bias->multiplier);
                sum2 += (int32_t)(input[2][j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);
            saturate_store_out(&sum2, &out[2], nSat);

            out += N;
        }
    }

    if (4 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);
                saturate(&sum2, nSat);
                saturate(&sum3, nSat);

                // kpartial = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                acc_iters = niters / (256 * SSE_16CAP);
                rem_iters = niters % (256 * SSE_16CAP);

                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();
                    acc2 = _mm_setzero_si128();
                    acc3 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr0 + ix);
                        in1 = _mm_load_si128(in_ptr1 + ix);
                        in2 = _mm_load_si128(in_ptr2 + ix);
                        in3 = _mm_load_si128(in_ptr3 + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                        weight += SSE_16CAP;

                        // multiply and add - won't saturate
                        in0 = _mm_madd_epi16(in0, w);
                        in1 = _mm_madd_epi16(in1, w);
                        in2 = _mm_madd_epi16(in2, w);
                        in3 = _mm_madd_epi16(in3, w);

                        acc0 = _mm_add_epi32(acc0, in0);
                        acc1 = _mm_add_epi32(acc1, in1);
                        acc2 = _mm_add_epi32(acc2, in2);
                        acc3 = _mm_add_epi32(acc3, in3);
                    }

                    sum0 += vec_sum32(acc0) * bias->multiplier;
                    sum1 += vec_sum32(acc1) * bias->multiplier;
                    sum2 += vec_sum32(acc2) * bias->multiplier;
                    sum3 += vec_sum32(acc3) * bias->multiplier;
                }

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();

                ix_end = ix + rem_iters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    in2 = _mm_load_si128(in_ptr2 + ix);
                    in3 = _mm_load_si128(in_ptr3 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    in1 = _mm_madd_epi16(in1, w);
                    in2 = _mm_madd_epi16(in2, w);
                    in3 = _mm_madd_epi16(in3, w);

                    acc0 = _mm_add_epi32(acc0, in0);
                    acc1 = _mm_add_epi32(acc1, in1);
                    acc2 = _mm_add_epi32(acc2, in2);
                    acc3 = _mm_add_epi32(acc3, in3);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum1 += (int32_t)(input[1][j] * *weight * bias->multiplier);
                sum2 += (int32_t)(input[2][j] * *weight * bias->multiplier);
                sum3 += (int32_t)(input[3][j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);
            saturate_store_out(&sum2, &out[2], nSat);
            saturate_store_out(&sum3, &out[3], nSat);

            out += N;
        }
    }

    if (5 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);
                saturate(&sum2, nSat);
                saturate(&sum3, nSat);
                saturate(&sum4, nSat);

                // kpartial = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;
                acc_iters = niters / (256 * SSE_16CAP);
                rem_iters = niters % (256 * SSE_16CAP);

                for (i = 0; i < acc_iters; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();
                    acc2 = _mm_setzero_si128();
                    acc3 = _mm_setzero_si128();
                    acc4 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr0 + ix);
                        in1 = _mm_load_si128(in_ptr1 + ix);
                        in2 = _mm_load_si128(in_ptr2 + ix);
                        in3 = _mm_load_si128(in_ptr3 + ix);
                        in4 = _mm_load_si128(in_ptr4 + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                        weight += SSE_16CAP;

                        // multiply and add - won't saturate
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

                    sum0 += vec_sum32(acc0) * bias->multiplier;
                    sum1 += vec_sum32(acc1) * bias->multiplier;
                    sum2 += vec_sum32(acc2) * bias->multiplier;
                    sum3 += vec_sum32(acc3) * bias->multiplier;
                    sum4 += vec_sum32(acc4) * bias->multiplier;
                }

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();
                acc4 = _mm_setzero_si128();

                ix_end = ix + rem_iters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    in2 = _mm_load_si128(in_ptr2 + ix);
                    in3 = _mm_load_si128(in_ptr3 + ix);
                    in4 = _mm_load_si128(in_ptr4 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
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

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
                sum4 += vec_sum32(acc4) * bias->multiplier;
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum1 += (int32_t)(input[1][j] * *weight * bias->multiplier);
                sum2 += (int32_t)(input[2][j] * *weight * bias->multiplier);
                sum3 += (int32_t)(input[3][j] * *weight * bias->multiplier);
                sum4 += (int32_t)(input[4][j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);
            saturate_store_out(&sum2, &out[2], nSat);
            saturate_store_out(&sum3, &out[3], nSat);
            saturate_store_out(&sum4, &out[4], nSat);

            out += N;
        }
    }

    if (6 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;
            sum5 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);
                saturate(&sum2, nSat);
                saturate(&sum3, nSat);
                saturate(&sum4, nSat);
                saturate(&sum5, nSat);

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 2016
                // 2016 / (8 * 256) < 1, acc won't saturate
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    in2 = _mm_load_si128(in_ptr2 + ix);
                    in3 = _mm_load_si128(in_ptr3 + ix);
                    in4 = _mm_load_si128(in_ptr4 + ix);
                    in5 = _mm_load_si128(in_ptr5 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
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

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
                sum4 += vec_sum32(acc4) * bias->multiplier;
                sum5 += vec_sum32(acc5) * bias->multiplier;

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();
                acc4 = _mm_setzero_si128();
                acc5 = _mm_setzero_si128();
            }

            for (j = 0; j < KT; j++, weight++)
            {
                sum0 += (int32_t)(input[0][j] * *weight * bias->multiplier);
                sum1 += (int32_t)(input[1][j] * *weight * bias->multiplier);
                sum2 += (int32_t)(input[2][j] * *weight * bias->multiplier);
                sum3 += (int32_t)(input[3][j] * *weight * bias->multiplier);
                sum4 += (int32_t)(input[4][j] * *weight * bias->multiplier);
                sum5 += (int32_t)(input[5][j] * *weight * bias->multiplier);
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);
            saturate_store_out(&sum2, &out[2], nSat);
            saturate_store_out(&sum3, &out[3], nSat);
            saturate_store_out(&sum4, &out[4], nSat);
            saturate_store_out(&sum5, &out[5], nSat);

            out += N;
        }
    }

    if (7 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;
            sum5 = bias->bias;
            sum6 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, nSat);
                saturate(&sum1, nSat);
                saturate(&sum2, nSat);
                saturate(&sum3, nSat);
                saturate(&sum4, nSat);
                saturate(&sum5, nSat);
                saturate(&sum6, nSat);

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 1728
                // 1728 / 256 = 6.75
                // 1728 / (8 * 256) < 1, acc won't saturate
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    in2 = _mm_load_si128(in_ptr2 + ix);
                    in3 = _mm_load_si128(in_ptr3 + ix);
                    in4 = _mm_load_si128(in_ptr4 + ix);
                    in5 = _mm_load_si128(in_ptr5 + ix);
                    in6 = _mm_load_si128(in_ptr6 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
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

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
                sum4 += vec_sum32(acc4) * bias->multiplier;
                sum5 += vec_sum32(acc5) * bias->multiplier;
                sum6 += vec_sum32(acc6) * bias->multiplier;

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();
                acc4 = _mm_setzero_si128();
                acc5 = _mm_setzero_si128();
                acc6 = _mm_setzero_si128();
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
            }

            saturate_store_out(&sum0, &out[0], nSat);
            saturate_store_out(&sum1, &out[1], nSat);
            saturate_store_out(&sum2, &out[2], nSat);
            saturate_store_out(&sum3, &out[3], nSat);
            saturate_store_out(&sum4, &out[4], nSat);
            saturate_store_out(&sum5, &out[5], nSat);
            saturate_store_out(&sum6, &out[6], nSat);

            out += N;
        }
    }

    if (8 == N)
    {
        for (l = 0; l < L; l++)
        {
            i = AL[l];
            weight = const_cast<int8_t*>(W)+i*K;
            bias = const_cast<nn_bias_c*>(B)+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;
            sum5 = bias->bias;
            sum6 = bias->bias;
            sum7 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();
            acc7 = _mm_setzero_si128();

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

                niters = kpartial < KK - kk * kpartial ? kpartial : KK - kk * kpartial;

                // kpartial = 1536
                // 1536 / 256 = 6
                // 1536 / (8 * 256) < 1, acc won't saturate
                ix_end = ix + niters / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr1 + ix);
                    in2 = _mm_load_si128(in_ptr2 + ix);
                    in3 = _mm_load_si128(in_ptr3 + ix);
                    in4 = _mm_load_si128(in_ptr4 + ix);
                    in5 = _mm_load_si128(in_ptr5 + ix);
                    in6 = _mm_load_si128(in_ptr6 + ix);
                    in7 = _mm_load_si128(in_ptr7 + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadu_si64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
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

                sum0 += vec_sum32(acc0) * bias->multiplier;
                sum1 += vec_sum32(acc1) * bias->multiplier;
                sum2 += vec_sum32(acc2) * bias->multiplier;
                sum3 += vec_sum32(acc3) * bias->multiplier;
                sum4 += vec_sum32(acc4) * bias->multiplier;
                sum5 += vec_sum32(acc5) * bias->multiplier;
                sum6 += vec_sum32(acc6) * bias->multiplier;
                sum7 += vec_sum32(acc7) * bias->multiplier;

                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();
                acc4 = _mm_setzero_si128();
                acc5 = _mm_setzero_si128();
                acc6 = _mm_setzero_si128();
                acc7 = _mm_setzero_si128();
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
igemm8_subset_mb(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int8_t*     W,
    const   nn_bias_s*  B,
    const   uint32_t    BG,
    const   nn_bias_c*  CB,
    int32_t*    O,
    const   uint32_t*   AL,
    const   uint32_t    L,
    uint32_t*   nSat,
    aligned_fv_bufs*    fvBuffers)
{}
