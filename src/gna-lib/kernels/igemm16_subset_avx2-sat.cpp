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

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

void AffineActiveListKernelImpl2B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t KT = config->inputElementCount % VEC_16CAP; // config->inputElementCount tail for manual processing
    uint32_t KK = config->inputElementCount - KT; // trimmed config->inputElementCount for AVX2 processing
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t niters;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;
    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    // simd inputs and weight
    __m256i in[8];
    __m256i w;

    // simd intermediate
    __m256i imm0;
    __m256i imm1;
    __m256i imm2;
    __m256i imm3;
    __m256i imm4;
    __m256i imm5;
    __m256i imm6;
    __m256i imm7;
    __m256i imm8;
    __m256i imm9;

    // simd accumulators
    __m256i acc[8];

    // simd input pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    int16_t const * input[8];
    memset(input, 0, sizeof(input));

    int64_t sum[8];            // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    int16_t const * weight;
    nn_bias_s const * bias;
    int32_t * output;

    output = config->output;

    if (1 == config->inputVectorCount)
    {
        in_ptr0 = (__m256i*)config->input;
        *input = config->input+KK;
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            *acc = _mm256_setzero_si256();
            *sum = *bias;
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                saturate(sum, config->execution->SaturationCount);

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
                *sum += (*input)[j] * *weight;
            }

            saturate_store_out(sum, output, config->execution->SaturationCount);

            output++;
        }
        return;
    }

    if (config->inputVectorCount == 8)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d7[i] = config->input[i*config->inputVectorCount + 7];
        }
        in_ptr7 = (__m256i*)config->execution->Intermediate->d7;
        input[7] = config->execution->Intermediate->d7 + KK;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        in_ptr6 = (__m256i*)config->execution->Intermediate->d6;
        input[6] = config->execution->Intermediate->d6 + KK;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        in_ptr5 = (__m256i*)config->execution->Intermediate->d5;
        input[5] = config->execution->Intermediate->d5 + KK;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        in_ptr4 = (__m256i*)config->execution->Intermediate->d4;
        input[4] = config->execution->Intermediate->d4 + KK;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        in_ptr3 = (__m256i*)config->execution->Intermediate->d3;
        input[3] = config->execution->Intermediate->d3 + KK;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        in_ptr2 = (__m256i*)config->execution->Intermediate->d2;
        input[2] = config->execution->Intermediate->d2 + KK;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        in_ptr1 = (__m256i*)config->execution->Intermediate->d1;
        input[1] = config->execution->Intermediate->d1 + KK;

        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        in_ptr0 = (__m256i*)config->execution->Intermediate->d0;
        input[0] = config->execution->Intermediate->d0 + KK;
    }

    if (2 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();

            sum[0] = *bias;
            sum[1] = *bias;

            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (3 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();

            sum[0] = *bias;
            sum[1] = *bias;
            sum[2] = *bias;
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
                sum[2] += input[2][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum[2], &output[2], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (4 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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
                sum[0] += input[0][j] * *weight;
                sum[1] += input[1][j] * *weight;
                sum[2] += input[2][j] * *weight;
                sum[3] += input[3][j] * *weight;
            }

            saturate_store_out(&sum[0], &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum[1], &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum[2], &output[2], config->execution->SaturationCount);
            saturate_store_out(&sum[3], &output[3], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (5 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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

            for (i = 0; i < config->inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->execution->SaturationCount);
            }

            output += config->inputVectorCount;
        }
    }

    if (6 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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

            for (i = 0; i < config->inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->execution->SaturationCount);
            }

            output += config->inputVectorCount;
        }
    }

    if (7 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm256_setzero_si256();
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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

            for (i = 0; i < config->inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->execution->SaturationCount);
            }

            output += config->inputVectorCount;
        }
    }

    if (8 == config->inputVectorCount)
    {
        for (l = 0; l < al->count; l++)
        {
            i = al->indices[l];
            weight = config->weights2B+i*config->inputElementCount;
            bias = config->biasesSimple+i;

            acc[0] = _mm256_setzero_si256();
            acc[1] = _mm256_setzero_si256();
            acc[2] = _mm256_setzero_si256();
            acc[3] = _mm256_setzero_si256();
            acc[4] = _mm256_setzero_si256();
            acc[5] = _mm256_setzero_si256();
            acc[6] = _mm256_setzero_si256();
            acc[7] = _mm256_setzero_si256();

            for (i = 0; i < config->inputVectorCount; i++)
            {
                sum[i] = *bias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm256_setzero_si256();
                    saturate(&sum[i], config->execution->SaturationCount);
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

            for (i = 0; i < config->inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < KT; j++, weight++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], output++, config->execution->SaturationCount);
            }
        }
    }
}
