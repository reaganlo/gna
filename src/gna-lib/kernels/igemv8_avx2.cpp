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

#include "igemv8.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <immintrin.h>
#include <cstdint>

void RecurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    uint32_t LDA = config->RequestConfig->Transform.outputElementCount + config->RequestConfig->Transform.inputElementCount;
    int16_t const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int16_t * feedback = config->RequestConfig->Transform.feedbackBuffer;

    int16_t const * const inputEnd = input + config->RequestConfig->Transform.inputElementCount - config->RequestConfig->Transform.inputElementCount % 16;
    int16_t const * const feedbackEnd = feedback + config->RequestConfig->Transform.outputElementCount - config->RequestConfig->Transform.outputElementCount % 16;

    BiasCompound const * bias = config->RequestConfig->Transform.biasesCompound;
    BiasCompound const * const biasEnd = bias + config->RequestConfig->Transform.outputElementCount;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Transform.output);
    int8_t const * weight = config->RequestConfig->Transform.weights1B;
    int8_t const * weight2 = weight + config->RequestConfig->Transform.inputElementCount;

    __m256i v0;
    __m256i v1;
    __m256i v2;

    for (; bias < biasEnd; bias++)
    {
        v2 = _mm256_setzero_si256();

        input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
        feedback = config->RequestConfig->Transform.feedbackBuffer;

        v0 = _mm256_lddqu_si256((__m256i*)input);
        v1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));

        while (input < inputEnd)
        {
            input += 16;
            weight += 16;

            v1 = _mm256_madd_epi16(v0, v1);
            v2 = _mm256_add_epi32(v1, v2);

            v0 = _mm256_lddqu_si256((__m256i*)input);
            v1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight));
        }

        v0 = _mm256_lddqu_si256((__m256i*)feedback);
        v1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight2));

        while (feedback < feedbackEnd)
        {
            feedback += 16;
            weight2 += 16;

            v1 = _mm256_madd_epi16(v0, v1);
            v2 = _mm256_add_epi32(v1, v2);

            v0 = _mm256_lddqu_si256((__m256i*)feedback);
            v1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128((__m128i*)weight2));
        }

        *output = vec_sum(v2);

        while (input < inputEnd + config->RequestConfig->Transform.inputElementCount % 16)
        {
            *output += *input++ * *weight++;
        }

        while (feedback < feedbackEnd + config->RequestConfig->Transform.outputElementCount % 16)
        {
            *output += *feedback++ * *weight2++;
        }

        *output = *output * bias->Multiplier + bias->Bias;
        output++;

        weight += LDA - config->RequestConfig->Transform.inputElementCount;
        weight2 += LDA - config->RequestConfig->Transform.outputElementCount;
    }
}
