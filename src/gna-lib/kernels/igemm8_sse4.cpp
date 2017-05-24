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

void AffineKernelImpl1B(AffineConfig const * const config)
{
    uint32_t i, j, ix, ix_end;
    uint32_t KT = config->inputElementCount % SSE_16CAP;
    uint32_t KK = config->inputElementCount - KT;
    ix_end = KK / SSE_16CAP;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;
    nn_bias_c const * bias;
    nn_bias_c const * const biasEnd = config->biasesCompound+config->outputElementCount;

    int16_t const * input_0, *input_1, *input_2, *input_3, *input_4, *input_5, *input_6, *input_7;
    __m128i *in_ptr0, *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4, *in_ptr5, *in_ptr6, *in_ptr7;
    __m128i in0, in1, in2, in3, in4, in5, in6, in7, w;
    __m128i acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m128i v0, v1, v2, v3, v4, v5, v6;

    ix_end = KK / SSE_16CAP;
    if (1 == config->inputVectorCount)
    {
        input_0 = config->input+KK;
        in_ptr0 = (__m128i*) config->input;
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
        {
            acc0 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            *output = bias->bias + vec_sum(acc0) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                *output += input_0[j] * *weight * bias->multiplier;
            }
            output++;
        }

        return;
    }

    switch (config->inputVectorCount)
    {
    case 8: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d7[i] = config->input[i*config->inputVectorCount + 7];
        input_7 = config->fvBuffers->d7 + KK;
        in_ptr7 = (__m128i*) config->fvBuffers->d7;
    case 7: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d6[i] = config->input[i*config->inputVectorCount + 6];
        input_6 = config->fvBuffers->d6 + KK;
        in_ptr6 = (__m128i*) config->fvBuffers->d6;
    case 6: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d5[i] = config->input[i*config->inputVectorCount + 5];
        input_5 = config->fvBuffers->d5 + KK;
        in_ptr5 = (__m128i*) config->fvBuffers->d5;
    case 5: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d4[i] = config->input[i*config->inputVectorCount + 4];
        input_4 = config->fvBuffers->d4 + KK;
        in_ptr4 = (__m128i*) config->fvBuffers->d4;
    case 4: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d3[i] = config->input[i*config->inputVectorCount + 3];
        input_3 = config->fvBuffers->d3 + KK;
        in_ptr3 = (__m128i*) config->fvBuffers->d3;
    case 3: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d2[i] = config->input[i*config->inputVectorCount + 2];
        input_2 = config->fvBuffers->d2 + KK;
        in_ptr2 = (__m128i*) config->fvBuffers->d2;
    case 2: 
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d1[i] = config->input[i*config->inputVectorCount + 1];
        input_1 = config->fvBuffers->d1 + KK;
        in_ptr1 = (__m128i*) config->fvBuffers->d1;
        for (i = 0; i < config->inputElementCount; i++) config->fvBuffers->d0[i] = config->input[i*config->inputVectorCount];
        input_0 = config->fvBuffers->d0 + KK;
        in_ptr0 = (__m128i*) config->fvBuffers->d0;
    }

    if (2 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }

    if (3 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
        {
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            for (ix = 0; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += SSE_16CAP;

                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }

    if (4 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
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

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
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

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }

    if (5 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
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

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
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

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;
            output[4] = bias->bias + vec_sum(acc4) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }

    if (6 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
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

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
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

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;
            output[4] = bias->bias + vec_sum(acc4) * bias->multiplier;
            output[5] = bias->bias + vec_sum(acc5) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
                output[5] += input_5[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }

    if (7 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
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

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
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

            output[0] = bias->bias + vec_sum(acc0) * bias->multiplier;
            output[1] = bias->bias + vec_sum(acc1) * bias->multiplier;
            output[2] = bias->bias + vec_sum(acc2) * bias->multiplier;
            output[3] = bias->bias + vec_sum(acc3) * bias->multiplier;
            output[4] = bias->bias + vec_sum(acc4) * bias->multiplier;
            output[5] = bias->bias + vec_sum(acc5) * bias->multiplier;
            output[6] = bias->bias + vec_sum(acc6) * bias->multiplier;

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
                output[5] += input_5[j] * *weight * bias->multiplier;
                output[6] += input_6[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }

    if (8 == config->inputVectorCount)
    {
        for (bias = config->biasesCompound; bias < biasEnd; bias++)
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
                in0 = _mm_load_si128(in_ptr0 + ix);
                in1 = _mm_load_si128(in_ptr1 + ix);
                in2 = _mm_load_si128(in_ptr2 + ix);
                in3 = _mm_load_si128(in_ptr3 + ix);
                in4 = _mm_load_si128(in_ptr4 + ix);
                in5 = _mm_load_si128(in_ptr5 + ix);
                in6 = _mm_load_si128(in_ptr6 + ix);
                in7 = _mm_load_si128(in_ptr7 + ix);

                w = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
                weight += SSE_16CAP;

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

            v0 = _mm_set1_epi32(bias->multiplier);
            v1 = _mm_set1_epi32(bias->bias);

            v2 = _mm_hadd_epi32(acc0, acc1);
            v3 = _mm_hadd_epi32(acc2, acc3);
            v4 = _mm_hadd_epi32(v2, v3);
            v5 = _mm_mullo_epi32(v4, v0);
            v6 = _mm_add_epi32(v5, v1);
            _mm_store_si128((__m128i*)output, v6);

            v2 = _mm_hadd_epi32(acc4, acc5);
            v3 = _mm_hadd_epi32(acc6, acc7);
            v4 = _mm_hadd_epi32(v2, v3);
            v5 = _mm_mullo_epi32(v4, v0);
            v6 = _mm_add_epi32(v5, v1);
            _mm_store_si128((__m128i*)(output + 4), v6);

            for (j = 0; j < KT; j++, weight++)
            {
                output[0] += input_0[j] * *weight * bias->multiplier;
                output[1] += input_1[j] * *weight * bias->multiplier;
                output[2] += input_2[j] * *weight * bias->multiplier;
                output[3] += input_3[j] * *weight * bias->multiplier;
                output[4] += input_4[j] * *weight * bias->multiplier;
                output[5] += input_5[j] * *weight * bias->multiplier;
                output[6] += input_6[j] * *weight * bias->multiplier;
                output[7] += input_7[j] * *weight * bias->multiplier;
            }
            output += config->inputVectorCount;
        }

        return;
    }
}

void AffineMultiBiasKernelImpl1B(AffineConfig const * const config)
{}
