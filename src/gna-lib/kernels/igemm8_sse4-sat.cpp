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

#include "KernelArguments.h"
#include "KernelMacros.h"
#include "Macros.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

static void initializeVectors(AffineConfig const *config,
        int16_t const * input[8], __m128i *in_ptr[8], uint32_t simdVectorLength);

void AffineKernelImpl1B(AffineConfig const * const config)
{
    uint32_t inputBufferSize;
    uint32_t vectorTailLength;
    uint32_t simdVectorLength;
    uint32_t numberOfIterationsPerGroup;
    uint32_t numberOfElementsPerGroup;
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;

    vectorTailLength = config->inputElementCount % SSE_16CAP; // config->inputElementCount tail for manual processing
    simdVectorLength = config->inputElementCount - vectorTailLength; // trimmed config->inputElementCount for AVX2 processing
    inputBufferSize = config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX];
    numberOfElementsPerGroup = inputBufferSize / config->inputVectorCount;
    numberOfIterationsPerGroup = config->inputElementCount / numberOfElementsPerGroup;

    int8_t const * weight;
    nn_bias_c const * bias = config->biasesCompound;;
    int32_t * output;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;
    output = config->output;
    weight = config->weights1B;

    // accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;
    int64_t sum6;
    int64_t sum7;

    // simd weights
    __m128i w0;
    __m128i w1;
    __m128i w;

    int16_t const * input[8];
    memset(input, 0, sizeof(input));

    // simd input pointers
    __m128i *in_ptr[8];
    memset(in_ptr, 0, sizeof(in_ptr));

    initializeVectors(config, input, in_ptr, simdVectorLength);

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i in7;

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
        input[0] = config->input + simdVectorLength;
        in_ptr[0] = (__m128i*)config->input;

        for (; bias < biasEnd; bias++)
        {
            ix = 0;
            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            sum0 = bias->bias;

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                acc0 = _mm_add_epi32(acc0, acc1);
                sum0 += vec_sum32(acc0) * bias->multiplier;

                saturate(&sum0, config->execution->SaturationCount);

                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
                numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
                remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

                // numberOfElementsPerGroup is 12288
                // 12288 / 256 = 48
                // max iters = 48 / SSE_16CAP = 6
                for (i = 0; i < numberOfSimdIterations; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix += 2)
                    {
                        in0 = _mm_load_si128(in_ptr[0] + ix);
                        in1 = _mm_load_si128(in_ptr[0] + ix + 1);

                        w0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                        w1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)(weight + SSE_16CAP)));
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

                ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);

                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    acc0 = _mm_add_epi32(acc0, in0);
                }

                sum0 += vec_sum32(acc0) * bias->multiplier;
                acc0 = _mm_setzero_si128();
            }

            for (j = 0; j < vectorTailLength; j++)
            {
                sum0 += (*input)[j] * *weight++ * bias->multiplier;
            }

            saturate_store_out(&sum0, output, config->execution->SaturationCount);

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
        input[7] = config->execution->Intermediate->d7 + simdVectorLength;
        in_ptr[7] = (__m128i*)config->execution->Intermediate->d7;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        input[6] = config->execution->Intermediate->d6 + simdVectorLength;
        in_ptr[6] = (__m128i*)config->execution->Intermediate->d6;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        input[5] = config->execution->Intermediate->d5 + simdVectorLength;
        in_ptr[5] = (__m128i*)config->execution->Intermediate->d5;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        input[4] = config->execution->Intermediate->d4 + simdVectorLength;
        in_ptr[4] = (__m128i*)config->execution->Intermediate->d4;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        input[3] = config->execution->Intermediate->d3 + simdVectorLength;
        in_ptr[3] = (__m128i*)config->execution->Intermediate->d3;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        input[2] = config->execution->Intermediate->d2 + simdVectorLength;
        in_ptr[2] = (__m128i*)config->execution->Intermediate->d2;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        input[1] = config->execution->Intermediate->d1 + simdVectorLength;
        in_ptr[1] = (__m128i*)config->execution->Intermediate->d1;

        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        input[0] = config->execution->Intermediate->d0 + simdVectorLength;
        in_ptr[0] = (__m128i*)config->execution->Intermediate->d0;
    }

    if (2 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);

                // numberOfElementsPerGroup = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
                numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
                remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

                for (i = 0; i < numberOfSimdIterations; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr[0] + ix);
                        in1 = _mm_load_si128(in_ptr[1] + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

                ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (3 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);
                saturate(&sum2, config->execution->SaturationCount);

                // numberOfElementsPerGroup = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup
                    ? numberOfElementsPerGroup
                    : simdVectorLength - (kk * numberOfElementsPerGroup);
                numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
                remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

                for (i = 0; i < numberOfSimdIterations; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();
                    acc2 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr[0] + ix);
                        in1 = _mm_load_si128(in_ptr[1] + ix);
                        in2 = _mm_load_si128(in_ptr[2] + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

                ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (4 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);
                saturate(&sum2, config->execution->SaturationCount);
                saturate(&sum3, config->execution->SaturationCount);

                // numberOfElementsPerGroup = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
                numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
                remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

                for (i = 0; i < numberOfSimdIterations; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();
                    acc2 = _mm_setzero_si128();
                    acc3 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr[0] + ix);
                        in1 = _mm_load_si128(in_ptr[1] + ix);
                        in2 = _mm_load_si128(in_ptr[2] + ix);
                        in3 = _mm_load_si128(in_ptr[3] + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

                ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
                sum3 += input[3][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
            saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (5 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
            ix = 0;

            sum0 = bias->bias;
            sum1 = bias->bias;
            sum2 = bias->bias;
            sum3 = bias->bias;
            sum4 = bias->bias;

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);
                saturate(&sum2, config->execution->SaturationCount);
                saturate(&sum3, config->execution->SaturationCount);
                saturate(&sum4, config->execution->SaturationCount);

                // numberOfElementsPerGroup = 12000 / 5 = 2400
                // 2016 / (8 * 256) = 1
                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
                numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
                remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

                for (i = 0; i < numberOfSimdIterations; i++)
                {
                    acc0 = _mm_setzero_si128();
                    acc1 = _mm_setzero_si128();
                    acc2 = _mm_setzero_si128();
                    acc3 = _mm_setzero_si128();
                    acc4 = _mm_setzero_si128();

                    ix_end = ix + 256;
                    for (; ix < ix_end; ix++)
                    {
                        in0 = _mm_load_si128(in_ptr[0] + ix);
                        in1 = _mm_load_si128(in_ptr[1] + ix);
                        in2 = _mm_load_si128(in_ptr[2] + ix);
                        in3 = _mm_load_si128(in_ptr[3] + ix);
                        in4 = _mm_load_si128(in_ptr[4] + ix);
                        w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

                ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    in4 = _mm_load_si128(in_ptr[4] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
                sum3 += input[3][j] * *weight * bias->multiplier;
                sum4 += input[4][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
            saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
            saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (6 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
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

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);
                saturate(&sum2, config->execution->SaturationCount);
                saturate(&sum3, config->execution->SaturationCount);
                saturate(&sum4, config->execution->SaturationCount);
                saturate(&sum5, config->execution->SaturationCount);

                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;

                // numberOfElementsPerGroup = 2016
                // 2016 / (8 * 256) < 1, acc won't saturate
                ix_end = ix + numberOfIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    in4 = _mm_load_si128(in_ptr[4] + ix);
                    in5 = _mm_load_si128(in_ptr[5] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
                sum3 += input[3][j] * *weight * bias->multiplier;
                sum4 += input[4][j] * *weight * bias->multiplier;
                sum5 += input[5][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
            saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
            saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);
            saturate_store_out(&sum5, &output[5], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (7 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
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

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);
                saturate(&sum2, config->execution->SaturationCount);
                saturate(&sum3, config->execution->SaturationCount);
                saturate(&sum4, config->execution->SaturationCount);
                saturate(&sum5, config->execution->SaturationCount);
                saturate(&sum6, config->execution->SaturationCount);

                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;

                // numberOfElementsPerGroup = 1728
                // 1728 / 256 = 6.75
                // 1728 / (8 * 256) < 1, acc won't saturate
                ix_end = ix + numberOfIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    in4 = _mm_load_si128(in_ptr[4] + ix);
                    in5 = _mm_load_si128(in_ptr[5] + ix);
                    in6 = _mm_load_si128(in_ptr[6] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
                sum3 += input[3][j] * *weight * bias->multiplier;
                sum4 += input[4][j] * *weight * bias->multiplier;
                sum5 += input[5][j] * *weight * bias->multiplier;
                sum6 += input[6][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
            saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
            saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);
            saturate_store_out(&sum5, &output[5], config->execution->SaturationCount);
            saturate_store_out(&sum6, &output[6], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }

    if (8 == config->inputVectorCount)
    {
        for (; bias < biasEnd; bias++)
        {
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

            for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
            {
                saturate(&sum0, config->execution->SaturationCount);
                saturate(&sum1, config->execution->SaturationCount);
                saturate(&sum2, config->execution->SaturationCount);
                saturate(&sum3, config->execution->SaturationCount);
                saturate(&sum4, config->execution->SaturationCount);
                saturate(&sum5, config->execution->SaturationCount);
                saturate(&sum6, config->execution->SaturationCount);
                saturate(&sum7, config->execution->SaturationCount);

                numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;

                // numberOfElementsPerGroup = 1536
                // 1536 / 256 = 6
                // 1536 / (8 * 256) < 1, acc won't saturate
                ix_end = ix + numberOfIterations / SSE_16CAP;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    in4 = _mm_load_si128(in_ptr[4] + ix);
                    in5 = _mm_load_si128(in_ptr[5] + ix);
                    in6 = _mm_load_si128(in_ptr[6] + ix);
                    in7 = _mm_load_si128(in_ptr[7] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            for (j = 0; j < vectorTailLength; j++, weight++)
            {
                sum0 += input[0][j] * *weight * bias->multiplier;
                sum1 += input[1][j] * *weight * bias->multiplier;
                sum2 += input[2][j] * *weight * bias->multiplier;
                sum3 += input[3][j] * *weight * bias->multiplier;
                sum4 += input[4][j] * *weight * bias->multiplier;
                sum5 += input[5][j] * *weight * bias->multiplier;
                sum6 += input[6][j] * *weight * bias->multiplier;
                sum7 += input[7][j] * *weight * bias->multiplier;
            }

            saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
            saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
            saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
            saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
            saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);
            saturate_store_out(&sum5, &output[5], config->execution->SaturationCount);
            saturate_store_out(&sum6, &output[6], config->execution->SaturationCount);
            saturate_store_out(&sum7, &output[7], config->execution->SaturationCount);

            output += config->inputVectorCount;
        }
    }
}

void affineMultiBiasKernelImpl1B_N1(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    UNREFERENCED_PARAMETER(input);
    UNREFERENCED_PARAMETER(in_ptr);

    uint32_t numberOfIterations;
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;

    // simd weights
    __m128i w;
    __m128i w0;
    __m128i w1;

    // simd input pointers
    __m128i *in_ptr0 = nullptr;

    // simd inputs
    __m128i in0;
    __m128i in1;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;

    // accumulators's sums
    int64_t sum;

    const int16_t *in = config->input + simdVectorLength;
    in_ptr0 = (__m128i*)config->input;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;
        acc0 = _mm_setzero_si128();
        acc1 = _mm_setzero_si128();
        sum = getBias(multiBias, config->bytesPerBias);

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            acc0 = _mm_add_epi32(acc0, acc1);
            sum += vec_sum32(acc0) * weightScaleFactor->multiplier;

            saturate(&sum, config->execution->SaturationCount);

            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
            numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
            remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

            // numberOfElementsPerGroup is 12288
            // 12288 / 256 = 48
            // max iters = 48 / SSE_16CAP = 6
            for (i = 0; i < numberOfSimdIterations; i++)
            {
                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();

                ix_end = ix + 256;
                for (; ix < ix_end; ix += 2)
                {
                    in0 = _mm_load_si128(in_ptr0 + ix);
                    in1 = _mm_load_si128(in_ptr0 + ix + 1);

                    w0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                    w1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)(weight + SSE_16CAP)));
                    weight += 2 * SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w0);
                    in1 = _mm_madd_epi16(in1, w1);
                    acc0 = _mm_add_epi32(acc0, in0);
                    acc1 = _mm_add_epi32(acc1, in1);
                }

                acc0 = _mm_add_epi32(acc0, acc1);
                sum += vec_sum32(acc0) * weightScaleFactor->multiplier;
            }

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr0 + ix);

                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                weight += SSE_16CAP;

                // multiply and add - won't saturate
                in0 = _mm_madd_epi16(in0, w);
                acc0 = _mm_add_epi32(acc0, in0);
            }

            sum += vec_sum32(acc0) * weightScaleFactor->multiplier;
            acc0 = _mm_setzero_si128();
        }

        for (j = 0; j < vectorTailLength; j++)
        {
            sum += in[j] * *weight++ * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);

        output++;
        weightScaleFactor++;
    }
}

static void affineMultiBiasKernelImpl1B_N2(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);

            // numberOfElementsPerGroup = 12000 / 5 = 2400
            // 2016 / (8 * 256) = 1
            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
            numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
            remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

            for (i = 0; i < numberOfSimdIterations; i++)
            {
                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();

                ix_end = ix + 256;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    in1 = _mm_madd_epi16(in1, w);

                    acc0 = _mm_add_epi32(acc0, in0);
                    acc1 = _mm_add_epi32(acc1, in1);
                }

                sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
                sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            }

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();

            ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                weight += SSE_16CAP;

                // multiply and add - won't saturate
                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
            }

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }
}

static void affineMultiBiasKernelImpl1B_N3(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);
        sum2 = getBias(multiBias, config->bytesPerBias);

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);
            saturate(&sum2, config->execution->SaturationCount);

            // numberOfElementsPerGroup = 12000 / 5 = 2400
            // 2016 / (8 * 256) = 1
            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
            numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
            remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

            for (i = 0; i < numberOfSimdIterations; i++)
            {
                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();

                ix_end = ix + 256;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                    weight += SSE_16CAP;

                    // multiply and add - won't saturate
                    in0 = _mm_madd_epi16(in0, w);
                    in1 = _mm_madd_epi16(in1, w);
                    in2 = _mm_madd_epi16(in2, w);

                    acc0 = _mm_add_epi32(acc0, in0);
                    acc1 = _mm_add_epi32(acc1, in1);
                    acc2 = _mm_add_epi32(acc2, in2);
                }

                sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
                sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
                sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
            }

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();

            ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                in2 = _mm_load_si128(in_ptr[2] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
                weight += SSE_16CAP;

                // multiply and add - won't saturate
                in0 = _mm_madd_epi16(in0, w);
                in1 = _mm_madd_epi16(in1, w);
                in2 = _mm_madd_epi16(in2, w);

                acc0 = _mm_add_epi32(acc0, in0);
                acc1 = _mm_add_epi32(acc1, in1);
                acc2 = _mm_add_epi32(acc2, in2);
            }

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
            sum2 += input[2][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }

}

static void affineMultiBiasKernelImpl1B_N4(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);
        sum2 = getBias(multiBias, config->bytesPerBias);
        sum3 = getBias(multiBias, config->bytesPerBias);

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);
            saturate(&sum2, config->execution->SaturationCount);
            saturate(&sum3, config->execution->SaturationCount);

            // numberOfElementsPerGroup = 12000 / 5 = 2400
            // 2016 / (8 * 256) = 1
            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;
            numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
            remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

            for (i = 0; i < numberOfSimdIterations; i++)
            {
                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();

                ix_end = ix + 256;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

                sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
                sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
                sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
                sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
            }

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();

            ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                in2 = _mm_load_si128(in_ptr[2] + ix);
                in3 = _mm_load_si128(in_ptr[3] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
            sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
            sum2 += input[2][j] * *weight * weightScaleFactor->multiplier;
            sum3 += input[3][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }

}

static void affineMultiBiasKernelImpl1B_N5(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t processedInputCount;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t i;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);
        sum2 = getBias(multiBias, config->bytesPerBias);
        sum3 = getBias(multiBias, config->bytesPerBias);
        sum4 = getBias(multiBias, config->bytesPerBias);

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);
            saturate(&sum2, config->execution->SaturationCount);
            saturate(&sum3, config->execution->SaturationCount);
            saturate(&sum4, config->execution->SaturationCount);

            // numberOfElementsPerGroup = 12000 / 5 = 2400
            // 2016 / (8 * 256) = 1
            processedInputCount = kk * numberOfElementsPerGroup;
            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - processedInputCount
                ? numberOfElementsPerGroup
                : simdVectorLength - processedInputCount;
            numberOfSimdIterations = numberOfIterations / (256 * SSE_16CAP);
            remainderOfSimdIterations = numberOfIterations % (256 * SSE_16CAP);

            for (i = 0; i < numberOfSimdIterations; i++)
            {
                acc0 = _mm_setzero_si128();
                acc1 = _mm_setzero_si128();
                acc2 = _mm_setzero_si128();
                acc3 = _mm_setzero_si128();
                acc4 = _mm_setzero_si128();

                ix_end = ix + 256;
                for (; ix < ix_end; ix++)
                {
                    in0 = _mm_load_si128(in_ptr[0] + ix);
                    in1 = _mm_load_si128(in_ptr[1] + ix);
                    in2 = _mm_load_si128(in_ptr[2] + ix);
                    in3 = _mm_load_si128(in_ptr[3] + ix);
                    in4 = _mm_load_si128(in_ptr[4] + ix);
                    w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

                sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
                sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
                sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
                sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
                sum4 += vec_sum32(acc4) * weightScaleFactor->multiplier;
            }

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();

            ix_end = ix + remainderOfSimdIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                in2 = _mm_load_si128(in_ptr[2] + ix);
                in3 = _mm_load_si128(in_ptr[3] + ix);
                in4 = _mm_load_si128(in_ptr[4] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
            sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
            sum4 += vec_sum32(acc4) * weightScaleFactor->multiplier;
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
            sum2 += input[2][j] * *weight * weightScaleFactor->multiplier;
            sum3 += input[3][j] * *weight * weightScaleFactor->multiplier;
            sum4 += input[4][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }

}

static void affineMultiBiasKernelImpl1B_N6(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);
        sum2 = getBias(multiBias, config->bytesPerBias);
        sum3 = getBias(multiBias, config->bytesPerBias);
        sum4 = getBias(multiBias, config->bytesPerBias);
        sum5 = getBias(multiBias, config->bytesPerBias);

        acc0 = _mm_setzero_si128();
        acc1 = _mm_setzero_si128();
        acc2 = _mm_setzero_si128();
        acc3 = _mm_setzero_si128();
        acc4 = _mm_setzero_si128();
        acc5 = _mm_setzero_si128();

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);
            saturate(&sum2, config->execution->SaturationCount);
            saturate(&sum3, config->execution->SaturationCount);
            saturate(&sum4, config->execution->SaturationCount);
            saturate(&sum5, config->execution->SaturationCount);

            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;

            // numberOfElementsPerGroup = 2016
            // 2016 / (8 * 256) < 1, acc won't saturate
            ix_end = ix + numberOfIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                in2 = _mm_load_si128(in_ptr[2] + ix);
                in3 = _mm_load_si128(in_ptr[3] + ix);
                in4 = _mm_load_si128(in_ptr[4] + ix);
                in5 = _mm_load_si128(in_ptr[5] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
            sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
            sum4 += vec_sum32(acc4) * weightScaleFactor->multiplier;
            sum5 += vec_sum32(acc5) * weightScaleFactor->multiplier;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
            sum2 += input[2][j] * *weight * weightScaleFactor->multiplier;
            sum3 += input[3][j] * *weight * weightScaleFactor->multiplier;
            sum4 += input[4][j] * *weight * weightScaleFactor->multiplier;
            sum5 += input[5][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);
        saturate_store_out(&sum5, &output[5], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }

}

static void affineMultiBiasKernelImpl1B_N7(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;
    int64_t sum6;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);
        sum2 = getBias(multiBias, config->bytesPerBias);
        sum3 = getBias(multiBias, config->bytesPerBias);
        sum4 = getBias(multiBias, config->bytesPerBias);
        sum5 = getBias(multiBias, config->bytesPerBias);
        sum6 = getBias(multiBias, config->bytesPerBias);

        acc0 = _mm_setzero_si128();
        acc1 = _mm_setzero_si128();
        acc2 = _mm_setzero_si128();
        acc3 = _mm_setzero_si128();
        acc4 = _mm_setzero_si128();
        acc5 = _mm_setzero_si128();
        acc6 = _mm_setzero_si128();

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);
            saturate(&sum2, config->execution->SaturationCount);
            saturate(&sum3, config->execution->SaturationCount);
            saturate(&sum4, config->execution->SaturationCount);
            saturate(&sum5, config->execution->SaturationCount);
            saturate(&sum6, config->execution->SaturationCount);

            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;

            // numberOfElementsPerGroup = 1728
            // 1728 / 256 = 6.75
            // 1728 / (8 * 256) < 1, acc won't saturate
            ix_end = ix + numberOfIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                in2 = _mm_load_si128(in_ptr[2] + ix);
                in3 = _mm_load_si128(in_ptr[3] + ix);
                in4 = _mm_load_si128(in_ptr[4] + ix);
                in5 = _mm_load_si128(in_ptr[5] + ix);
                in6 = _mm_load_si128(in_ptr[6] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
            sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
            sum4 += vec_sum32(acc4) * weightScaleFactor->multiplier;
            sum5 += vec_sum32(acc5) * weightScaleFactor->multiplier;
            sum6 += vec_sum32(acc6) * weightScaleFactor->multiplier;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
            sum2 += input[2][j] * *weight * weightScaleFactor->multiplier;
            sum3 += input[3][j] * *weight * weightScaleFactor->multiplier;
            sum4 += input[4][j] * *weight * weightScaleFactor->multiplier;
            sum5 += input[5][j] * *weight * weightScaleFactor->multiplier;
            sum6 += input[6][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);
        saturate_store_out(&sum5, &output[5], config->execution->SaturationCount);
        saturate_store_out(&sum6, &output[6], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }

}

static void affineMultiBiasKernelImpl1B_N8(
    AffineConfig const * const config, int16_t const *input[8], __m128i *in_ptr[8],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t j;
    uint32_t kk;

    int8_t const *weight;
    int32_t * output;
    auto const * multiBias = static_cast<int8_t const *>(config->multiBias);
    auto const * const biasEnd = static_cast<int8_t const *>(config->multiBias) +
        (config->bytesPerBias * config->outputElementCount * config->multiBiasVectorCount);
    auto biasStride = config->bytesPerBias * config->multiBiasVectorCount;
    nn_scaling const * weightScaleFactor = config->weightScaleFactors;

    output = config->output;
    weight = config->weights1B;
    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;
    __m128i in2;
    __m128i in3;
    __m128i in4;
    __m128i in5;
    __m128i in6;
    __m128i in7;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;
    __m128i acc2;
    __m128i acc3;
    __m128i acc4;
    __m128i acc5;
    __m128i acc6;
    __m128i acc7;

    // accumulators's sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;
    int64_t sum6;
    int64_t sum7;

    for (; multiBias < biasEnd; multiBias += biasStride)
    {
        ix = 0;

        sum0 = getBias(multiBias, config->bytesPerBias);
        sum1 = getBias(multiBias, config->bytesPerBias);
        sum2 = getBias(multiBias, config->bytesPerBias);
        sum3 = getBias(multiBias, config->bytesPerBias);
        sum4 = getBias(multiBias, config->bytesPerBias);
        sum5 = getBias(multiBias, config->bytesPerBias);
        sum6 = getBias(multiBias, config->bytesPerBias);
        sum7 = getBias(multiBias, config->bytesPerBias);

        acc0 = _mm_setzero_si128();
        acc1 = _mm_setzero_si128();
        acc2 = _mm_setzero_si128();
        acc3 = _mm_setzero_si128();
        acc4 = _mm_setzero_si128();
        acc5 = _mm_setzero_si128();
        acc6 = _mm_setzero_si128();
        acc7 = _mm_setzero_si128();

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->execution->SaturationCount);
            saturate(&sum1, config->execution->SaturationCount);
            saturate(&sum2, config->execution->SaturationCount);
            saturate(&sum3, config->execution->SaturationCount);
            saturate(&sum4, config->execution->SaturationCount);
            saturate(&sum5, config->execution->SaturationCount);
            saturate(&sum6, config->execution->SaturationCount);
            saturate(&sum7, config->execution->SaturationCount);

            numberOfIterations = numberOfElementsPerGroup < simdVectorLength - kk * numberOfElementsPerGroup ? numberOfElementsPerGroup : simdVectorLength - kk * numberOfElementsPerGroup;

            // numberOfElementsPerGroup = 1536
            // 1536 / 256 = 6
            // 1536 / (8 * 256) < 1, acc won't saturate
            ix_end = ix + numberOfIterations / SSE_16CAP;
            for (; ix < ix_end; ix++)
            {
                in0 = _mm_load_si128(in_ptr[0] + ix);
                in1 = _mm_load_si128(in_ptr[1] + ix);
                in2 = _mm_load_si128(in_ptr[2] + ix);
                in3 = _mm_load_si128(in_ptr[3] + ix);
                in4 = _mm_load_si128(in_ptr[4] + ix);
                in5 = _mm_load_si128(in_ptr[5] + ix);
                in6 = _mm_load_si128(in_ptr[6] + ix);
                in7 = _mm_load_si128(in_ptr[7] + ix);
                w = _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)weight));
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

            sum0 += vec_sum32(acc0) * weightScaleFactor->multiplier;
            sum1 += vec_sum32(acc1) * weightScaleFactor->multiplier;
            sum2 += vec_sum32(acc2) * weightScaleFactor->multiplier;
            sum3 += vec_sum32(acc3) * weightScaleFactor->multiplier;
            sum4 += vec_sum32(acc4) * weightScaleFactor->multiplier;
            sum5 += vec_sum32(acc5) * weightScaleFactor->multiplier;
            sum6 += vec_sum32(acc6) * weightScaleFactor->multiplier;
            sum7 += vec_sum32(acc7) * weightScaleFactor->multiplier;

            acc0 = _mm_setzero_si128();
            acc1 = _mm_setzero_si128();
            acc2 = _mm_setzero_si128();
            acc3 = _mm_setzero_si128();
            acc4 = _mm_setzero_si128();
            acc5 = _mm_setzero_si128();
            acc6 = _mm_setzero_si128();
            acc7 = _mm_setzero_si128();
        }

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += input[0][j] * *weight * weightScaleFactor->multiplier;
            sum1 += input[1][j] * *weight * weightScaleFactor->multiplier;
            sum2 += input[2][j] * *weight * weightScaleFactor->multiplier;
            sum3 += input[3][j] * *weight * weightScaleFactor->multiplier;
            sum4 += input[4][j] * *weight * weightScaleFactor->multiplier;
            sum5 += input[5][j] * *weight * weightScaleFactor->multiplier;
            sum6 += input[6][j] * *weight * weightScaleFactor->multiplier;
            sum7 += input[7][j] * *weight * weightScaleFactor->multiplier;
        }

        saturate_store_out(&sum0, &output[0], config->execution->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->execution->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->execution->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->execution->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->execution->SaturationCount);
        saturate_store_out(&sum5, &output[5], config->execution->SaturationCount);
        saturate_store_out(&sum6, &output[6], config->execution->SaturationCount);
        saturate_store_out(&sum7, &output[7], config->execution->SaturationCount);

        output += config->inputVectorCount;
        weightScaleFactor++;
    }

}

void AffineMultiBiasKernelImpl1B(AffineConfig const * const config)
{
    uint32_t vectorTailLength;
    uint32_t simdVectorLength;
    uint32_t inputBufferSize;
    uint32_t numberOfElementsPerGroup;
    uint32_t numberOfIterationsPerGroup;

    inputBufferSize = config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX];
    numberOfElementsPerGroup = inputBufferSize / config->inputVectorCount;
    numberOfIterationsPerGroup = config->inputElementCount / numberOfElementsPerGroup;

    vectorTailLength = config->inputElementCount % SSE_16CAP; // config->inputElementCount tail for manual processing
    simdVectorLength = config->inputElementCount - vectorTailLength; // trimmed config->inputElementCount for AVX2 processing

    __m128i *in_ptr[8];
    memset(in_ptr, 0, sizeof(in_ptr));

    int16_t const *input[8];
    memset(input, 0, sizeof(input));

    initializeVectors(config, input, in_ptr, simdVectorLength);

    switch(config->inputVectorCount)
    {
        case 1:
            affineMultiBiasKernelImpl1B_N1(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 2:
            affineMultiBiasKernelImpl1B_N2(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 3:
            affineMultiBiasKernelImpl1B_N3(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 4:
            affineMultiBiasKernelImpl1B_N4(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 5:
            affineMultiBiasKernelImpl1B_N5(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 6:
            affineMultiBiasKernelImpl1B_N6(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 7:
            affineMultiBiasKernelImpl1B_N7(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
        case 8:
            affineMultiBiasKernelImpl1B_N8(config, input, in_ptr, numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    }
}

void initializeVectors(AffineConfig const * const config,
        int16_t const * input[8], __m128i *in_ptr[8], uint32_t simdVectorLength)
{
    uint32_t i;

    if (config->inputVectorCount == 8)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d7[i] = config->input[i*config->inputVectorCount + 7];
        }
        input[7] = config->execution->Intermediate->d7 + simdVectorLength;
        in_ptr[7] = (__m128i*)config->execution->Intermediate->d7;
    }
    if (config->inputVectorCount >= 7)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d6[i] = config->input[i*config->inputVectorCount + 6];
        }
        input[6] = config->execution->Intermediate->d6 + simdVectorLength;
        in_ptr[6] = (__m128i*)config->execution->Intermediate->d6;
    }
    if (config->inputVectorCount >= 6)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d5[i] = config->input[i*config->inputVectorCount + 5];
        }
        input[5] = config->execution->Intermediate->d5 + simdVectorLength;
        in_ptr[5] = (__m128i*)config->execution->Intermediate->d5;
    }
    if (config->inputVectorCount >= 5)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d4[i] = config->input[i*config->inputVectorCount + 4];
        }
        input[4] = config->execution->Intermediate->d4 + simdVectorLength;
        in_ptr[4] = (__m128i*)config->execution->Intermediate->d4;
    }
    if (config->inputVectorCount >= 4)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d3[i] = config->input[i*config->inputVectorCount + 3];
        }
        input[3] = config->execution->Intermediate->d3 + simdVectorLength;
        in_ptr[3] = (__m128i*)config->execution->Intermediate->d3;
    }
    if (config->inputVectorCount >= 3)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d2[i] = config->input[i*config->inputVectorCount + 2];
        }
        input[2] = config->execution->Intermediate->d2 + simdVectorLength;
        in_ptr[2] = (__m128i*)config->execution->Intermediate->d2;
    }
    if (config->inputVectorCount >= 2)
    {
        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d1[i] = config->input[i*config->inputVectorCount + 1];
        }
        input[1] = config->execution->Intermediate->d1 + simdVectorLength;
        in_ptr[1] = (__m128i*)config->execution->Intermediate->d1;

        for (i = 0; i < config->inputElementCount; i++)
        {
            config->execution->Intermediate->d0[i] = config->input[i*config->inputVectorCount];
        }
        input[0] = config->execution->Intermediate->d0 + simdVectorLength;
        in_ptr[0] = (__m128i*)config->execution->Intermediate->d0;
    }
}

