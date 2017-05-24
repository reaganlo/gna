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

#include <string.h>

#include "igemv.h"
#include "igemv8.h"

void AffineActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t i, j, k, l, kk, kpartial, nKpartial, niters;
    kpartial = (hw_buf_size[config->inputVectorCount - 1]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    __m128i in[8], w, w0, w1;     // inputs & weight
    __m128i imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7, imm8, imm9, imm10;       // immediate
    __m128i acc[8]; // output accumulators
    __m128i zero = _mm_setzero_si128(); // AVX2 ZERO

    int16_t const * input[8];
    int8_t const * weight;             
    nn_bias_c const * bias = config->biasesCompound;
    int32_t * output;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;
    int64_t sum[8];            // 64-bit accumulator buffer

    uint32_t KT = config->inputElementCount % SSE_16CAP; // config->inputElementCount tail for manual processing
    uint32_t KK = config->inputElementCount - KT; // trimmed config->inputElementCount for AVX2 processing

    output = config->output;
    weight = config->weights1B;

    __m128i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    __m128i in0, in1, in2, in3, in4, in5, in6, in7;
    __m128i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    int64_t sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    uint32_t ix, ix_end;

    if (1 == config->inputVectorCount)
    {
        in_ptr0 = (__m128i*)config->input;
        input[0] = config->input+KK;
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;

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
                saturate_store_out(&sum0, output, config->saturationCount);
                sum0 = *output;

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

            saturate_store_out(&sum0, output, config->saturationCount);

            output++;
        }
        return;
    }

    switch (config->inputVectorCount)
    {
    case 8: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d7[i] = config->input[i*config->inputVectorCount + 7];
        input[7] = config->fvBuffers->d7 + KK;
        in_ptr7 = (__m128i*)config->fvBuffers->d7;
    case 7: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d6[i] = config->input[i*config->inputVectorCount + 6];
        input[6] = config->fvBuffers->d6 + KK;
        in_ptr6 = (__m128i*)config->fvBuffers->d6;
    case 6: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d5[i] = config->input[i*config->inputVectorCount + 5];
        input[5] = config->fvBuffers->d5 + KK;
        in_ptr5 = (__m128i*)config->fvBuffers->d5;
    case 5: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d4[i] = config->input[i*config->inputVectorCount + 4];
        input[4] = config->fvBuffers->d4 + KK;
        in_ptr4 = (__m128i*)config->fvBuffers->d4;
    case 4: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d3[i] = config->input[i*config->inputVectorCount + 3];
        input[3] = config->fvBuffers->d3 + KK;
        in_ptr3 = (__m128i*)config->fvBuffers->d3;
    case 3: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d2[i] = config->input[i*config->inputVectorCount + 2];
        input[2] = config->fvBuffers->d2 + KK;
        in_ptr2 = (__m128i*)config->fvBuffers->d2;
    case 2: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d1[i] = config->input[i*config->inputVectorCount + 1];
        input[1] = config->fvBuffers->d1 + KK;
        in_ptr1 = (__m128i*)config->fvBuffers->d1;
        
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d0[i] = config->input[i*config->inputVectorCount];
        input[0] = config->fvBuffers->d0 + KK;
        in_ptr0 = (__m128i*)config->fvBuffers->d0;
    }

    if (2 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);

            output += config->inputVectorCount;
        }
    }

    if (3 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);
                saturate(&sum2, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);
            saturate_store_out(&sum2, &output[2], config->saturationCount);

            output += config->inputVectorCount;
        }
    }

    if (4 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
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
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);
                saturate(&sum2, config->saturationCount);
                saturate(&sum3, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);
            saturate_store_out(&sum2, &output[2], config->saturationCount);
            saturate_store_out(&sum3, &output[3], config->saturationCount);

            output += config->inputVectorCount;
        }
    }

    if (5 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
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
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);
                saturate(&sum2, config->saturationCount);
                saturate(&sum3, config->saturationCount);
                saturate(&sum4, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);
            saturate_store_out(&sum2, &output[2], config->saturationCount);
            saturate_store_out(&sum3, &output[3], config->saturationCount);
            saturate_store_out(&sum4, &output[4], config->saturationCount);

            output += config->inputVectorCount;
        }
    }

    if (6 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
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
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);
                saturate(&sum2, config->saturationCount);
                saturate(&sum3, config->saturationCount);
                saturate(&sum4, config->saturationCount);
                saturate(&sum5, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);
            saturate_store_out(&sum2, &output[2], config->saturationCount);
            saturate_store_out(&sum3, &output[3], config->saturationCount);
            saturate_store_out(&sum4, &output[4], config->saturationCount);
            saturate_store_out(&sum5, &output[5], config->saturationCount);

            output += config->inputVectorCount;
        }
    }

    if (7 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
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
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);
                saturate(&sum2, config->saturationCount);
                saturate(&sum3, config->saturationCount);
                saturate(&sum4, config->saturationCount);
                saturate(&sum5, config->saturationCount);
                saturate(&sum6, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);
            saturate_store_out(&sum2, &output[2], config->saturationCount);
            saturate_store_out(&sum3, &output[3], config->saturationCount);
            saturate_store_out(&sum4, &output[4], config->saturationCount);
            saturate_store_out(&sum5, &output[5], config->saturationCount);
            saturate_store_out(&sum6, &output[6], config->saturationCount);

            output += config->inputVectorCount;
        }
    }

    if (8 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights1B+i*config->inputElementCount;
            bias = config->biasesCompound+i;
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
                saturate(&sum0, config->saturationCount);
                saturate(&sum1, config->saturationCount);
                saturate(&sum2, config->saturationCount);
                saturate(&sum3, config->saturationCount);
                saturate(&sum4, config->saturationCount);
                saturate(&sum5, config->saturationCount);
                saturate(&sum6, config->saturationCount);
                saturate(&sum7, config->saturationCount);

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

            saturate_store_out(&sum0, &output[0], config->saturationCount);
            saturate_store_out(&sum1, &output[1], config->saturationCount);
            saturate_store_out(&sum2, &output[2], config->saturationCount);
            saturate_store_out(&sum3, &output[3], config->saturationCount);
            saturate_store_out(&sum4, &output[4], config->saturationCount);
            saturate_store_out(&sum5, &output[5], config->saturationCount);
            saturate_store_out(&sum6, &output[6], config->saturationCount);
            saturate_store_out(&sum7, &output[7], config->saturationCount);

            output += config->inputVectorCount;
        }
    }
}

void AffineMultiBiasActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al)
{}
