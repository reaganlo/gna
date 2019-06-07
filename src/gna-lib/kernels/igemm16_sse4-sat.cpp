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

#include <cstring>
#include <immintrin.h>

void AffineKernelImpl2B(AffineConfig const * const config)
{
    // loop variables
    uint32_t KT = config->inputElementCount % SSE_16CAP; // config->inputElementCount tail for manual processing
    uint32_t KK = config->inputElementCount - KT; // trimmed config->inputElementCount for AVX2 processing
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t niters;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;

    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    // simd inputs and weight
    __m128i in[8];
    __m128i w;

    // simmd accumulators
    __m128i acc[8];

    // simd input pointers
    __m128i *in_ptr0 = nullptr;
    __m128i *in_ptr1 = nullptr;
    __m128i *in_ptr2 = nullptr;
    __m128i *in_ptr3 = nullptr;
    __m128i *in_ptr4 = nullptr;
    __m128i *in_ptr5 = nullptr;
    __m128i *in_ptr6 = nullptr;
    __m128i *in_ptr7 = nullptr;

    int16_t const * input[8];
    memset(input, 0, sizeof(input));

    void const * const bias = config->biasesSimple;
    int16_t const * weight;
    int32_t * output;

    int64_t sum[8];            // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    output = config->output;
    weight = config->weights2B;

    if (1 == config->inputVectorCount)
    {
        *input = config->input+KK;
        in_ptr0 = (__m128i*)config->input;
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            *acc = _mm_setzero_si128();
            *sum = getBias(bias, biasIndex, config->bytesPerBias);
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                *sum += vec_sum(*acc);
                *acc = _mm_setzero_si128();
                saturate(sum, config->execution->SaturationCount);

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
        in_ptr7 = (__m128i*)config->execution->Intermediate->d7;
        input[7] = config->execution->Intermediate->d7 + KK;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        in_ptr6 = (__m128i*)config->execution->Intermediate->d6;
        input[6] = config->execution->Intermediate->d6 + KK;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        in_ptr5 = (__m128i*)config->execution->Intermediate->d5;
        input[5] = config->execution->Intermediate->d5 + KK;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        in_ptr4 = (__m128i*)config->execution->Intermediate->d4;
        input[4] = config->execution->Intermediate->d4 + KK;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        in_ptr3 = (__m128i*)config->execution->Intermediate->d3;
        input[3] = config->execution->Intermediate->d3 + KK;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        in_ptr2 = (__m128i*)config->execution->Intermediate->d2;
        input[2] = config->execution->Intermediate->d2 + KK;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        in_ptr1 = (__m128i*)config->execution->Intermediate->d1;
        input[1] = config->execution->Intermediate->d1 + KK;

        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        in_ptr0 = (__m128i*)config->execution->Intermediate->d0;
        input[0] = config->execution->Intermediate->d0 + KK;
    }

    if (2 == config->inputVectorCount)
    {
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

    if (3 == config->inputVectorCount)
    {
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

    if (4 == config->inputVectorCount)
    {
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

    if (5 == config->inputVectorCount)
    {
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
        for (uint32_t biasIndex = 0; biasIndex < config->outputElementCount; biasIndex++)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, biasIndex, config->bytesPerBias);
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

                for (i = 0; i < config->inputVectorCount; i++)
                {
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

void AffineMultiBiasKernelImpl2B(AffineConfig const * const config)
{
    // loop variables
    uint32_t KT = config->inputElementCount % SSE_16CAP; // config->inputElementCount tail for manual processing
    uint32_t KK = config->inputElementCount - KT; // trimmed config->inputElementCount for AVX2 processing
    uint32_t kpartial;
    uint32_t nKpartial;
    uint32_t niters;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    // simd inputs and weight
    __m128i in[8];
    __m128i w;

    // simd accumulators
    __m128i acc[8];

    // simd input pointers
    __m128i *in_ptr0 = nullptr;
    __m128i *in_ptr1 = nullptr;
    __m128i *in_ptr2 = nullptr;
    __m128i *in_ptr3 = nullptr;
    __m128i *in_ptr4 = nullptr;
    __m128i *in_ptr5 = nullptr;
    __m128i *in_ptr6 = nullptr;
    __m128i *in_ptr7 = nullptr;

    int16_t const * input[8];
    memset(input, 0, sizeof(input));

    int16_t const * weight;
    int32_t * output;
    nn_bias_s const * multiBias = config->multiBias;
    nn_bias_s const * const biasEnd = multiBias + config->outputElementCount * config->multiBiasVectorCount;

    int64_t sum[8]; // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    output = config->output;
    weight = config->weights2B;

    if (1 == config->inputVectorCount)
    {
        *input = config->input+KK;
        in_ptr0 = (__m128i*)config->input;
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            *acc = _mm_setzero_si128();
            *sum = *multiBias;
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                *sum += vec_sum(*acc);
                *acc = _mm_setzero_si128();
                saturate(sum, config->execution->SaturationCount);

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
        in_ptr7 = (__m128i*)config->execution->Intermediate->d7;
        input[7] = config->execution->Intermediate->d7 + KK;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        in_ptr6 = (__m128i*)config->execution->Intermediate->d6;
        input[6] = config->execution->Intermediate->d6 + KK;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        in_ptr5 = (__m128i*)config->execution->Intermediate->d5;
        input[5] = config->execution->Intermediate->d5 + KK;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        in_ptr4 = (__m128i*)config->execution->Intermediate->d4;
        input[4] = config->execution->Intermediate->d4 + KK;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        in_ptr3 = (__m128i*)config->execution->Intermediate->d3;
        input[3] = config->execution->Intermediate->d3 + KK;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        in_ptr2 = (__m128i*)config->execution->Intermediate->d2;
        input[2] = config->execution->Intermediate->d2 + KK;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        in_ptr1 = (__m128i*)config->execution->Intermediate->d1;
        input[1] = config->execution->Intermediate->d1 + KK;

        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        in_ptr0 = (__m128i*)config->execution->Intermediate->d0;
        input[0] = config->execution->Intermediate->d0 + KK;
    }

    if (2 == config->inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

    if (3 == config->inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

    if (4 == config->inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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

    if (5 == config->inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
            }
            ix = 0;

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                for (i = 0; i < config->inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
        for (; multiBias < biasEnd; multiBias+=config->multiBiasVectorCount)
        {
            for (i = 0; i < config->inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = *multiBias;
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

                for (i = 0; i < config->inputVectorCount; i++)
                {
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->execution->SaturationCount);
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
