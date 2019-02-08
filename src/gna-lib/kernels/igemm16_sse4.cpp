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
#include "igemv16.h"

void AffineKernelImpl2B(AffineConfig const * const config)
{
    uint32_t KT = config->inputElementCount % SSE_16CAP;
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

    // simd input pointers
    __m128i *in_ptr0 = nullptr;
    __m128i *in_ptr1 = nullptr;
    __m128i *in_ptr2 = nullptr;
    __m128i *in_ptr3 = nullptr;
    __m128i *in_ptr4 = nullptr;
    __m128i *in_ptr5 = nullptr;
    __m128i *in_ptr6 = nullptr;
    __m128i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i w;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    uint8_t const * bias = (uint8_t*)config->biasesSimple;
    uint8_t const * const biasEnd = bias + (config->outputElementCount*config->bytesPerBias);

    if (1 == config->inputVectorCount)
    {
        in_ptr0 = (__m128i*)config->input;
        input_0 = config->input+KK;
        ix_end = KK / SSE_16CAP;
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            *output = vec_sum(acc0) + (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias);
            for (i = 0; i < KT; i++, weight++)
            {
                *output += input_0[i] * *weight;
            }
            output++;
        }

        return;
    }

    switch (config->inputVectorCount)
    {
    case 8:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d7[i] = config->input[i*config->inputVectorCount + 7];
        in_ptr7 = (__m128i*)config->execution->Intermediate->d7;
        input_7 = config->execution->Intermediate->d7 + KK;
    case 7:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        in_ptr6 = (__m128i*)config->execution->Intermediate->d6;
        input_6 = config->execution->Intermediate->d6 + KK;
    case 6:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        in_ptr5 = (__m128i*)config->execution->Intermediate->d5;
        input_5 = config->execution->Intermediate->d5 + KK;
    case 5:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        in_ptr4 = (__m128i*)config->execution->Intermediate->d4;
        input_4 = config->execution->Intermediate->d4 + KK;
    case 4:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        in_ptr3 = (__m128i*)config->execution->Intermediate->d3;
        input_3 = config->execution->Intermediate->d3 + KK;
    case 3:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        in_ptr2 = (__m128i*)config->execution->Intermediate->d2;
        input_2 = config->execution->Intermediate->d2 + KK;
    case 2:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        in_ptr1 = (__m128i*)config->execution->Intermediate->d1;
        input_1 = config->execution->Intermediate->d1 + KK;
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        in_ptr0 = (__m128i*)config->execution->Intermediate->d0;
        input_0 = config->execution->Intermediate->d0 + KK;
    }
    ix_end = KK / SSE_16CAP;

    if (2 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);

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
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);

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
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
            }

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);

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
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);

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
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);

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

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);
            output[5] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc5);

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
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);
                in6 = _mm_load_si128(in_ptr6 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);
            output[5] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc5);
            output[6] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc6);

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
        for (; bias < biasEnd; bias += config->bytesPerBias)
        {
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
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_load_si128(in_ptr0 + ix), w));
                acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_load_si128(in_ptr1 + ix), w));
                acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_load_si128(in_ptr2 + ix), w));
                acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_load_si128(in_ptr3 + ix), w));
                acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_load_si128(in_ptr4 + ix), w));
                acc5 = _mm_add_epi32(acc5, _mm_madd_epi16(_mm_load_si128(in_ptr5 + ix), w));
                acc6 = _mm_add_epi32(acc6, _mm_madd_epi16(_mm_load_si128(in_ptr6 + ix), w));
                acc7 = _mm_add_epi32(acc7, _mm_madd_epi16(_mm_load_si128(in_ptr7 + ix), w));
            }

            output[0] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);
            output[5] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc5);
            output[6] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc6);
            output[7] = (int32_t)getBias((void*)bias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc7);

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
            output += config->inputVectorCount;
        }

        return;
    }
}

void AffineMultiBiasKernelImpl2B(AffineConfig const * const config)
{
    uint32_t KT = config->inputElementCount % SSE_16CAP;
    uint32_t KK = config->inputElementCount - KT;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;

    uint8_t const * multiBias;
    uint8_t const * const biasEnd = (uint8_t*)config->multiBias + config->outputElementCount * config->multiBiasVectorCount * config->bytesPerBias;
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

    // simd input pointers
    __m128i *in_ptr0 = nullptr;
    __m128i *in_ptr1 = nullptr;
    __m128i *in_ptr2 = nullptr;
    __m128i *in_ptr3 = nullptr;
    __m128i *in_ptr4 = nullptr;
    __m128i *in_ptr5 = nullptr;
    __m128i *in_ptr6 = nullptr;
    __m128i *in_ptr7 = nullptr;

    // simd inputs and weight
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i w;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    if (1 == config->inputVectorCount)
    {
        in_ptr0 = (__m128i*)config->input;
        input_0 = config->input+KK;
        ix_end = KK / SSE_16CAP;
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            *output = vec_sum(acc0) + (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias);
            for (i = 0; i < KT; i++, weight++)
            {
                *output += input_0[i] * *weight;
            }
            output++;
        }

        return;
    }

    switch (config->inputVectorCount)
    {
    case 8:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d7[i] = config->input[i*config->inputVectorCount + 7];
        in_ptr7 = (__m128i*)config->execution->Intermediate->d7;
        input_7 = config->execution->Intermediate->d7 + KK;
    case 7:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        in_ptr6 = (__m128i*)config->execution->Intermediate->d6;
        input_6 = config->execution->Intermediate->d6 + KK;
    case 6:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        in_ptr5 = (__m128i*)config->execution->Intermediate->d5;
        input_5 = config->execution->Intermediate->d5 + KK;
    case 5:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        in_ptr4 = (__m128i*)config->execution->Intermediate->d4;
        input_4 = config->execution->Intermediate->d4 + KK;
    case 4:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        in_ptr3 = (__m128i*)config->execution->Intermediate->d3;
        input_3 = config->execution->Intermediate->d3 + KK;
    case 3:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        in_ptr2 = (__m128i*)config->execution->Intermediate->d2;
        input_2 = config->execution->Intermediate->d2 + KK;
    case 2:
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        in_ptr1 = (__m128i*)config->execution->Intermediate->d1;
        input_1 = config->execution->Intermediate->d1 + KK;
        for (i = 0; i < config->inputElementCount; i++) config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        in_ptr0 = (__m128i*)config->execution->Intermediate->d0;
        input_0 = config->execution->Intermediate->d0 + KK;
    }
    ix_end = KK / SSE_16CAP;

    if (2 == config->inputVectorCount)
    {
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);

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
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);

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
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);
                in3 = _mm_madd_epi16(in3, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
                acc3 = _mm_add_epi32(acc3, in3);
            }

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);

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
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);

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
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);

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

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);
            output[5] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc5);

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
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);
                in6 = _mm_load_si128(in_ptr6 + ix);

                w = _mm_lddqu_si128((__m128i*)weight);
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

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);
            output[5] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc5);
            output[6] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc6);

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
        for (multiBias = (uint8_t*)config->multiBias; multiBias < biasEnd; multiBias+=(config->multiBiasVectorCount*config->bytesPerBias))
        {
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
                w = _mm_lddqu_si128((__m128i*)weight);
                weight += SSE_16CAP;
                acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_load_si128(in_ptr0 + ix), w));
                acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_load_si128(in_ptr1 + ix), w));
                acc2 = _mm_add_epi32(acc2, _mm_madd_epi16(_mm_load_si128(in_ptr2 + ix), w));
                acc3 = _mm_add_epi32(acc3, _mm_madd_epi16(_mm_load_si128(in_ptr3 + ix), w));
                acc4 = _mm_add_epi32(acc4, _mm_madd_epi16(_mm_load_si128(in_ptr4 + ix), w));
                acc5 = _mm_add_epi32(acc5, _mm_madd_epi16(_mm_load_si128(in_ptr5 + ix), w));
                acc6 = _mm_add_epi32(acc6, _mm_madd_epi16(_mm_load_si128(in_ptr6 + ix), w));
                acc7 = _mm_add_epi32(acc7, _mm_madd_epi16(_mm_load_si128(in_ptr7 + ix), w));
            }

            output[0] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc0);
            output[1] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc1);
            output[2] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc2);
            output[3] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc3);
            output[4] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc4);
            output[5] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc5);
            output[6] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc6);
            output[7] = (int32_t)getBias((void*)multiBias, 0, (gna_data_mode)config->bytesPerBias) + vec_sum(acc7);

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
            output += config->inputVectorCount;
        }

        return;
    }
}
