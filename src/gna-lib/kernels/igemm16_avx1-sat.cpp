/*
 INTEL CONFIDENTIAL
 Copyright 2017-2021 Intel Corporation.

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

#include "saturate.h"
#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"


#include <cstdint>
#include <cstring>
#include <immintrin.h>

// TODO: make all kernel implementations style consistent, use auto and sort local variables, use full names, 1 var. per decl. line

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    auto weight = config->RequestConfig->Transform.weights2B;
    auto const *bias = reinterpret_cast<int8_t const *>(config->RequestConfig->Transform.biasesSimple);
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);

    auto const biasEnd = bias + (config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount);
    auto const tailElementCount = config->RequestConfig->Transform.inputElementCount % SSE_16CAP; // config->RequestConfig->Transform.inputElementCount tail for manual processing
    auto const headElementCount = config->RequestConfig->Transform.inputElementCount - tailElementCount; // trimmed config->RequestConfig->Transform.inputElementCount for AVX2 processing
    auto const partialCapacity = (config->BufferElementCount[config->RequestConfig->Transform.inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->RequestConfig->Transform.inputVectorCount;
    auto const iterationCount = config->RequestConfig->Transform.inputElementCount / partialCapacity;

    uint32_t iterationCapacity;
    uint32_t iter;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;

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

    int64_t sum[8]; // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        *input = inputs + headElementCount;
        in_ptr0 = (__m128i*)config->RequestConfig->Inputs;
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            *acc = _mm_setzero_si128();
            *sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                *sum += vec_sum(*acc);
                *acc = _mm_setzero_si128();
                saturate(sum, config->SaturationCount);

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                *sum += (*input)[j] * *weight;
            }

            saturate_store_out(sum, output, config->SaturationCount);

            output++;
        }
        return;
    }

    if (config->RequestConfig->Transform.inputVectorCount == 8)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d7[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 7];
        }
        in_ptr7 = (__m128i*)config->Intermediate->d7;
        input[7] = config->Intermediate->d7 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        in_ptr6 = (__m128i*)config->Intermediate->d6;
        input[6] = config->Intermediate->d6 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        in_ptr5 = (__m128i*)config->Intermediate->d5;
        input[5] = config->Intermediate->d5 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        in_ptr4 = (__m128i*)config->Intermediate->d4;
        input[4] = config->Intermediate->d4 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        in_ptr3 = (__m128i*)config->Intermediate->d3;
        input[3] = config->Intermediate->d3 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        in_ptr2 = (__m128i*)config->Intermediate->d2;
        input[2] = config->Intermediate->d2 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        in_ptr1 = (__m128i*)config->Intermediate->d1;
        input[1] = config->Intermediate->d1 + headElementCount;
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        in_ptr0 = (__m128i*)config->Intermediate->d0;
        input[0] = config->Intermediate->d0 + headElementCount;
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->RequestConfig->Transform.bytesPerBias)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(bias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                sum[0] += vec_sum(acc[0]);
                sum[1] += vec_sum(acc[1]);
                sum[2] += vec_sum(acc[2]);
                sum[3] += vec_sum(acc[3]);
                sum[4] += vec_sum(acc[4]);
                sum[5] += vec_sum(acc[5]);
                sum[6] += vec_sum(acc[6]);
                sum[7] += vec_sum(acc[7]);

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], output++, config->SaturationCount);
            }
        }
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    auto weight = config->RequestConfig->Transform.weights2B;
    auto multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias);
    auto output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    auto const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);

    auto const biasEnd = multiBias + config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.multiBiasVectorCount;
    auto const biasStride = config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount;
    auto const tailElementCount = config->RequestConfig->Transform.inputElementCount % SSE_16CAP; // config->RequestConfig->Transform.inputElementCount tail for manual processing
    auto const headElementCount = config->RequestConfig->Transform.inputElementCount - tailElementCount; // trimmed config->RequestConfig->Transform.inputElementCount for AVX2 processing
    auto const partialCapacity = (config->BufferElementCount[config->RequestConfig->Transform.inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->RequestConfig->Transform.inputVectorCount;
    auto const iterationCount = config->RequestConfig->Transform.inputElementCount / partialCapacity;

    uint32_t iterationCapacity;
    uint32_t iter;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;

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

    int64_t sum[8]; // 64-bit accumulator buffer
    memset(sum, 0, sizeof(sum));

    if (1 == config->RequestConfig->Transform.inputVectorCount)
    {
        *input = inputs + headElementCount;
        in_ptr0 = (__m128i*)config->RequestConfig->Inputs;
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            *acc = _mm_setzero_si128();
            *sum = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                *sum += vec_sum(*acc);
                *acc = _mm_setzero_si128();
                saturate(sum, config->SaturationCount);

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                *sum += (*input)[j] * *weight;
            }

            saturate_store_out(sum, output, config->SaturationCount);

            output++;
        }
        return;
    }

    if (config->RequestConfig->Transform.inputVectorCount == 8)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d7[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 7];
        }
        in_ptr7 = (__m128i*)config->Intermediate->d7;
        input[7] = config->Intermediate->d7 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        in_ptr6 = (__m128i*)config->Intermediate->d6;
        input[6] = config->Intermediate->d6 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        in_ptr5 = (__m128i*)config->Intermediate->d5;
        input[5] = config->Intermediate->d5 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        in_ptr4 = (__m128i*)config->Intermediate->d4;
        input[4] = config->Intermediate->d4 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        in_ptr3 = (__m128i*)config->Intermediate->d3;
        input[3] = config->Intermediate->d3 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        in_ptr2 = (__m128i*)config->Intermediate->d2;
        input[2] = config->Intermediate->d2 + headElementCount;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        in_ptr1 = (__m128i*)config->Intermediate->d1;
        input[1] = config->Intermediate->d1 + headElementCount;
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        in_ptr0 = (__m128i*)config->Intermediate->d0;
        input[0] = config->Intermediate->d0 + headElementCount;
    }

    if (2 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (3 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (4 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (5 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (6 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (7 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += vec_sum(acc[i]);
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], &output[i], config->SaturationCount);
            }

            output += config->RequestConfig->Transform.inputVectorCount;
        }
    }

    if (8 == config->RequestConfig->Transform.inputVectorCount)
    {
        for (; multiBias < biasEnd; multiBias += biasStride)
        {
            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                acc[i] = _mm_setzero_si128();
                sum[i] = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);
            }
            ix = 0;

            for (iter = 0; iter < iterationCount + 1; iter++)
            {
                sum[0] += vec_sum(acc[0]);
                sum[1] += vec_sum(acc[1]);
                sum[2] += vec_sum(acc[2]);
                sum[3] += vec_sum(acc[3]);
                sum[4] += vec_sum(acc[4]);
                sum[5] += vec_sum(acc[5]);
                sum[6] += vec_sum(acc[6]);
                sum[7] += vec_sum(acc[7]);

                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    acc[i] = _mm_setzero_si128();
                    saturate(&sum[i], config->SaturationCount);
                }

                iterationCapacity = partialCapacity < headElementCount - iter * partialCapacity ? partialCapacity : headElementCount - iter * partialCapacity;
                ix_end = ix + iterationCapacity / SSE_16CAP;
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

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                sum[i] += vec_sum(acc[i]);
            }

            for (j = 0; j < tailElementCount; j++, weight++)
            {
                for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
                {
                    sum[i] += input[i][j] * *weight;
                }
            }

            for (i = 0; i < config->RequestConfig->Transform.inputVectorCount; i++)
            {
                saturate_store_out(&sum[i], output++, config->SaturationCount);
            }
        }
    }
}

