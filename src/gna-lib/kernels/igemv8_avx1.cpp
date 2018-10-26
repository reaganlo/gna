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

void RecurrentKernelImpl1B(RecurrentConfig const * const config)
{
    uint32_t LDA = config->outputElementCount + config->inputElementCount;
    int16_t const * input = config->input;
    int16_t * feedback = config->feedbackBuffer;

    int16_t const * const inputEnd = input + config->inputElementCount - config->inputElementCount % 16;
    int16_t const * const feedbackEnd = feedback + config->outputElementCount - config->outputElementCount % 16;

    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;
    int32_t * output = config->output;
    int8_t const * weight = config->weights1B;
    int8_t const * weight2 = weight + config->inputElementCount;

    __m256i v0;
    __m128i s0;
    __m128i s1;
    __m128i s2;
    __m128i s3;
    __m128i s4;
    __m128i s5;

    for (; bias < biasEnd; bias++)
    {
        input = config->input;
        feedback = config->feedbackBuffer;

        v0 = _mm256_lddqu_si256((__m256i*)input);

        s0 = _mm256_castsi256_si128(v0);
        s1 = _mm256_extractf128_si256(v0, 1);

        s2 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
        s3 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)(weight + 8)));

        s5 = _mm_setzero_si128();

        while (input < inputEnd)
        {
            input += 16;
            weight += 16;

            s0 = _mm_madd_epi16(s0, s2);
            s1 = _mm_madd_epi16(s1, s3);

            s4 = _mm_add_epi32(s0, s1);
            s5 = _mm_add_epi32(s4, s5);

            v0 = _mm256_lddqu_si256((__m256i*)input);

            s0 = _mm256_castsi256_si128(v0);
            s1 = _mm256_extractf128_si256(v0, 1);

            s2 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
            s3 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)(weight + 8)));
        }

        v0 = _mm256_lddqu_si256((__m256i*)feedback);
        s0 = _mm256_castsi256_si128(v0);
        s1 = _mm256_extractf128_si256(v0, 1);
        s2 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight2));
        s3 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)(weight2 + 8)));

        while (feedback < feedbackEnd)
        {
            feedback += 16;
            weight2 += 16;

            s0 = _mm_madd_epi16(s0, s2);
            s1 = _mm_madd_epi16(s1, s3);

            s4 = _mm_add_epi32(s0, s1);
            s5 = _mm_add_epi32(s4, s5);

            v0 = _mm256_lddqu_si256((__m256i*)feedback);
            s0 = _mm256_castsi256_si128(v0);
            s1 = _mm256_extractf128_si256(v0, 1);
            s2 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight2));
            s3 = _mm_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)(weight2 + 8)));
        }

        *output = vec_sum(s5);

        while (input < inputEnd + config->inputElementCount % 16)
        {
            *output += *input++ * *weight++;
        }

        while (feedback < feedbackEnd + config->outputElementCount % 16)
        {
            *output += *feedback++ * *weight2++;
        }
        *output = *output * bias->multiplier + bias->bias;
        output++;

        weight += LDA - config->inputElementCount;
        weight2 += LDA - config->outputElementCount;
    }
}
