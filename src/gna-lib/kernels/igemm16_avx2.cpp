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

#include "igemv16.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"

#include <cstdint>
#include <immintrin.h>

void AffineKernelImpl2B(AffineConfig const * const config)
{
    uint32_t KT = config->inputElementCount % VEC_16CAP;
    uint32_t KK = config->inputElementCount - KT;
    uint32_t ix_end;
    uint32_t i;
    uint32_t ix;

    int32_t * output = config->output;
    int16_t const * weight = config->weights2B;
    int16_t const *input_0 = nullptr;
    int16_t const *input_1 = nullptr;
    int16_t const *input_2 = nullptr;
    int16_t const *input_3 = nullptr;
    int16_t const *input_4 = nullptr;
    int16_t const *input_5 = nullptr;
    int16_t const *input_6 = nullptr;
    int16_t const *input_7 = nullptr;

    // simd inputs pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m256i v0;
    __m256i v1;
    __m256i v2;
    __m256i v3;
    __m256i v4;
    __m256i v5;
    __m256i v6;
    __m256i in0;
    __m256i in1;
    __m256i in2;
    __m256i in3;
    __m256i in4;
    __m256i in5;
    __m256i in6;
    __m256i w;

    // simd accumulators
    __m256i acc0;
    __m256i acc1;
    __m256i acc2;
    __m256i acc3;
    __m256i acc4;
    __m256i acc5;
    __m256i acc6;
    __m256i acc7;
    __m128i s1;
    __m128i s2;
    __m128i s3;

    int8_t const *bias;
    auto const * const biasEnd = reinterpret_cast<int8_t const *>(config->biasesSimple) +
        (config->bytesPerBias * config->outputElementCount);

    if (1 == config->inputVectorCount)
    {
        in_ptr0 = (__m256i*)config->input;
        input_0 = config->input+KK;
        ix_end = KK / VEC_16CAP;
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                acc0 = _mm256_add_epi32(acc0, in0);
            }

            *output = vec_sum(acc0) + getBias(bias, config->bytesPerBias);
            for (i = 0; i < KT; i++, weight++)
            {
                *output += input_0[i] * *weight;
            }
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
        input_7 = config->execution->Intermediate->d7 + KK;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        in_ptr6 = (__m256i*)config->execution->Intermediate->d6;
        input_6 = config->execution->Intermediate->d6 + KK;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        in_ptr5 = (__m256i*)config->execution->Intermediate->d5;
        input_5 = config->execution->Intermediate->d5 + KK;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        in_ptr4 = (__m256i*)config->execution->Intermediate->d4;
        input_4 = config->execution->Intermediate->d4 + KK;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        in_ptr3 = (__m256i*)config->execution->Intermediate->d3;
        input_3 = config->execution->Intermediate->d3 + KK;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        in_ptr2 = (__m256i*)config->execution->Intermediate->d2;
        input_2 = config->execution->Intermediate->d2 + KK;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        in_ptr1 = (__m256i*)config->execution->Intermediate->d1;
        input_1 = config->execution->Intermediate->d1 + KK;
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        in_ptr0 = (__m256i*)config->execution->Intermediate->d0;
        input_0 = config->execution->Intermediate->d0 + KK;
    }
    ix_end = KK / VEC_16CAP;

    if (2 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
            }

            output[0] = getBias(bias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->bytesPerBias) + vec_sum(acc1);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (3 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
            }

            output[0] = getBias(bias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->bytesPerBias) + vec_sum(acc2);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (4 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
            }

            output[0] = getBias(bias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->bytesPerBias) + vec_sum(acc3);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (5 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
            }

            output[0] = getBias(bias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->bytesPerBias) + vec_sum(acc4);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (6 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
            }

            output[0] = getBias(bias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(bias, config->bytesPerBias) + vec_sum(acc5);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (7 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);
                in6 = _mm256_load_si256(in_ptr6 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);
                in6 = _mm256_madd_epi16(in6, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
                acc6 = _mm256_add_epi32(acc6, in6);
            }

            output[0] = getBias(bias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(bias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(bias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(bias, config->bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(bias, config->bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(bias, config->bytesPerBias) + vec_sum(acc5);
            output[6] = getBias(bias, config->bytesPerBias) + vec_sum(acc6);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (8 == config->inputVectorCount)
    {
        for (bias = reinterpret_cast<int8_t const *>(config->biasesSimple); bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();
            acc7 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_load_si256(in_ptr0 + ix), w));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_load_si256(in_ptr1 + ix), w));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_load_si256(in_ptr2 + ix), w));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_load_si256(in_ptr3 + ix), w));
                acc4 = _mm256_add_epi32(acc4, _mm256_madd_epi16(_mm256_load_si256(in_ptr4 + ix), w));
                acc5 = _mm256_add_epi32(acc5, _mm256_madd_epi16(_mm256_load_si256(in_ptr5 + ix), w));
                acc6 = _mm256_add_epi32(acc6, _mm256_madd_epi16(_mm256_load_si256(in_ptr6 + ix), w));
                acc7 = _mm256_add_epi32(acc7, _mm256_madd_epi16(_mm256_load_si256(in_ptr7 + ix), w));
            }

            v0 = _mm256_hadd_epi32(acc0, acc1);
            v1 = _mm256_hadd_epi32(acc2, acc3);
            v2 = _mm256_hadd_epi32(v0, v1);

            v3 = _mm256_hadd_epi32(acc4, acc5);
            v4 = _mm256_hadd_epi32(acc6, acc7);
            v5 = _mm256_hadd_epi32(v3, v4);

            s1 = _mm_set1_epi32(getBias(bias, config->bytesPerBias));
            s2 = _mm_add_epi32(_mm256_castsi256_si128(v2), _mm256_extracti128_si256(v2, 1));
            s2 = _mm_add_epi32(s1, s2);
            s3 = _mm_add_epi32(_mm256_castsi256_si128(v5), _mm256_extracti128_si256(v5, 1));
            s3 = _mm_add_epi32(s1, s3);

            v6 = _mm256_set_m128i(s3, s2);

            _mm256_store_si256((__m256i*)output, v6);
            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
                output[7] += input_7[i] * *weight;
            }
            output += 8;
        }

        return;
    }
}

void AffineMultiBiasKernelImpl2B(AffineConfig const * const config)
{
    uint32_t KT = config->inputElementCount % VEC_16CAP;
    uint32_t KK = config->inputElementCount - KT;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;

    int32_t * output = config->output;
    int16_t const * weight = config->weights2B;
    int16_t const *input_0 = nullptr;
    int16_t const *input_1 = nullptr;
    int16_t const *input_2 = nullptr;
    int16_t const *input_3 = nullptr;
    int16_t const *input_4 = nullptr;
    int16_t const *input_5 = nullptr;
    int16_t const *input_6 = nullptr;
    int16_t const *input_7 = nullptr;

    // simd inputs pointers
    __m256i *in_ptr0 = nullptr;
    __m256i *in_ptr1 = nullptr;
    __m256i *in_ptr2 = nullptr;
    __m256i *in_ptr3 = nullptr;
    __m256i *in_ptr4 = nullptr;
    __m256i *in_ptr5 = nullptr;
    __m256i *in_ptr6 = nullptr;
    __m256i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m256i v0;
    __m256i v1;
    __m256i v2;
    __m256i v3;
    __m256i v4;
    __m256i v5;
    __m256i v6;
    __m256i in0;
    __m256i in1;
    __m256i in2;
    __m256i in3;
    __m256i in4;
    __m256i in5;
    __m256i in6;
    __m256i w;

    // simd accumulators
    __m256i acc0;
    __m256i acc1;
    __m256i acc2;
    __m256i acc3;
    __m256i acc4;
    __m256i acc5;
    __m256i acc6;
    __m256i acc7;
    __m128i s1;
    __m128i s2;
    __m128i s3;

    int8_t const * multiBias;
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;

    if (1 == config->inputVectorCount)
    {
        in_ptr0 = (__m256i*)config->input;
        input_0 = config->input+KK;
        ix_end = KK / VEC_16CAP;
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                acc0 = _mm256_add_epi32(acc0, in0);
            }

            *output = vec_sum(acc0) + getBias(multiBias, config->bytesPerBias);
            for (i = 0; i < KT; i++, weight++)
            {
                *output += input_0[i] * *weight;
            }
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
        input_7 = config->execution->Intermediate->d7 + KK;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        in_ptr6 = (__m256i*)config->execution->Intermediate->d6;
        input_6 = config->execution->Intermediate->d6 + KK;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        in_ptr5 = (__m256i*)config->execution->Intermediate->d5;
        input_5 = config->execution->Intermediate->d5 + KK;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        in_ptr4 = (__m256i*)config->execution->Intermediate->d4;
        input_4 = config->execution->Intermediate->d4 + KK;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        in_ptr3 = (__m256i*)config->execution->Intermediate->d3;
        input_3 = config->execution->Intermediate->d3 + KK;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        in_ptr2 = (__m256i*)config->execution->Intermediate->d2;
        input_2 = config->execution->Intermediate->d2 + KK;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        in_ptr1 = (__m256i*)config->execution->Intermediate->d1;
        input_1 = config->execution->Intermediate->d1 + KK;
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        in_ptr0 = (__m256i*)config->execution->Intermediate->d0;
        input_0 = config->execution->Intermediate->d0 + KK;
    }
    ix_end = KK / VEC_16CAP;

    if (2 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
            }

            output[0] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc1);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (3 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
            }

            output[0] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc2);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (4 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
            }

            output[0] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc3);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (5 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
            }

            output[0] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc4);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (6 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
            }

            output[0] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc5);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (7 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm256_load_si256(in_ptr0 + ix);
                in1 = _mm256_load_si256(in_ptr1 + ix);
                in2 = _mm256_load_si256(in_ptr2 + ix);
                in3 = _mm256_load_si256(in_ptr3 + ix);
                in4 = _mm256_load_si256(in_ptr4 + ix);
                in5 = _mm256_load_si256(in_ptr5 + ix);
                in6 = _mm256_load_si256(in_ptr6 + ix);

                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;

                in0 = _mm256_madd_epi16(in0, w);
                in1 = _mm256_madd_epi16(in1, w);
                in2 = _mm256_madd_epi16(in2, w);
                in3 = _mm256_madd_epi16(in3, w);
                in4 = _mm256_madd_epi16(in4, w);
                in5 = _mm256_madd_epi16(in5, w);
                in6 = _mm256_madd_epi16(in6, w);

                acc0 = _mm256_add_epi32(acc0, in0);
                acc1 = _mm256_add_epi32(acc1, in1);
                acc2 = _mm256_add_epi32(acc2, in2);
                acc3 = _mm256_add_epi32(acc3, in3);
                acc4 = _mm256_add_epi32(acc4, in4);
                acc5 = _mm256_add_epi32(acc5, in5);
                acc6 = _mm256_add_epi32(acc6, in6);
            }

            output[0] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc0);
            output[1] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc1);
            output[2] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc2);
            output[3] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc3);
            output[4] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc4);
            output[5] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc5);
            output[6] = getBias(multiBias, config->bytesPerBias) + vec_sum(acc6);

            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
            }

            output += config->inputVectorCount;
        }

        return;
    }

    if (8 == config->inputVectorCount)
    {
        for (multiBias = static_cast<int8_t const *>(config->multiBias); multiBias < biasEnd; multiBias += biasStride)
        {
            acc0 = _mm256_setzero_si256();
            acc1 = _mm256_setzero_si256();
            acc2 = _mm256_setzero_si256();
            acc3 = _mm256_setzero_si256();
            acc4 = _mm256_setzero_si256();
            acc5 = _mm256_setzero_si256();
            acc6 = _mm256_setzero_si256();
            acc7 = _mm256_setzero_si256();

            for (ix = 0; ix < ix_end; ix++)
            {
                w = _mm256_lddqu_si256((__m256i*)weight);
                weight += VEC_16CAP;
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_load_si256(in_ptr0 + ix), w));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_load_si256(in_ptr1 + ix), w));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_load_si256(in_ptr2 + ix), w));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_load_si256(in_ptr3 + ix), w));
                acc4 = _mm256_add_epi32(acc4, _mm256_madd_epi16(_mm256_load_si256(in_ptr4 + ix), w));
                acc5 = _mm256_add_epi32(acc5, _mm256_madd_epi16(_mm256_load_si256(in_ptr5 + ix), w));
                acc6 = _mm256_add_epi32(acc6, _mm256_madd_epi16(_mm256_load_si256(in_ptr6 + ix), w));
                acc7 = _mm256_add_epi32(acc7, _mm256_madd_epi16(_mm256_load_si256(in_ptr7 + ix), w));
            }

            v0 = _mm256_hadd_epi32(acc0, acc1);
            v1 = _mm256_hadd_epi32(acc2, acc3);
            v2 = _mm256_hadd_epi32(v0, v1);

            v3 = _mm256_hadd_epi32(acc4, acc5);
            v4 = _mm256_hadd_epi32(acc6, acc7);
            v5 = _mm256_hadd_epi32(v3, v4);

            s1 = _mm_set1_epi32(getBias(multiBias, config->bytesPerBias));
            s2 = _mm_add_epi32(_mm256_castsi256_si128(v2), _mm256_extracti128_si256(v2, 1));
            s2 = _mm_add_epi32(s1, s2);
            s3 = _mm_add_epi32(_mm256_castsi256_si128(v5), _mm256_extracti128_si256(v5, 1));
            s3 = _mm_add_epi32(s1, s3);

            v6 = _mm256_set_m128i(s3, s2);

            _mm256_store_si256((__m256i*)output, v6);
            for (i = 0; i < KT; i++, weight++)
            {
                output[0] += input_0[i] * *weight;
                output[1] += input_1[i] * *weight;
                output[2] += input_2[i] * *weight;
                output[3] += input_3[i] * *weight;
                output[4] += input_4[i] * *weight;
                output[5] += input_5[i] * *weight;
                output[6] += input_6[i] * *weight;
                output[7] += input_7[i] * *weight;
            }
            output += 8;
        }

        return;
    }
}
