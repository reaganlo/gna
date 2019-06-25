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

#include "common.h"
#include "gna-api-types-xnn.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>

static void initializeVectors(ExecutionKernelConfig<AffineConfig> const *config,
        int16_t const * input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX], uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N1(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N2(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N3(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N4(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N5(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N6(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N7(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

static void affineActiveListKernelImpl1B_N8(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength);

void AffineActiveListKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t inputBufferSize;
    uint32_t vectorTailLength; // config->RequestConfig->Transform.inputElementCount tail for manual processing
    uint32_t simdVectorLength; // trimmed config->RequestConfig->Transform.inputElementCount for AVX2 processing
    uint32_t numberOfIterationsPerGroup;
    uint32_t numberOfElementsPerGroup;

    inputBufferSize = config->BufferElementCount[config->RequestConfig->Transform.inputVectorCount - 1 + XNN_N_GROUP_MAX];
    vectorTailLength = config->RequestConfig->Transform.inputElementCount % SSE_16CAP;
    simdVectorLength = config->RequestConfig->Transform.inputElementCount - vectorTailLength;
    numberOfElementsPerGroup = inputBufferSize / config->RequestConfig->Transform.inputVectorCount;
    numberOfIterationsPerGroup = config->RequestConfig->Transform.inputElementCount / numberOfElementsPerGroup;

    int16_t const * input[XNN_N_GROUP_MAX];
    memset(input, 0, sizeof(input));

    // simd input pointers
    __m128i *in_ptr[XNN_N_GROUP_MAX];
    memset(in_ptr, 0, sizeof(in_ptr));

    initializeVectors(config, input, in_ptr, simdVectorLength);

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 1:
            affineActiveListKernelImpl1B_N1(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 2:
            affineActiveListKernelImpl1B_N2(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 3:
            affineActiveListKernelImpl1B_N3(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 4:
            affineActiveListKernelImpl1B_N4(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 5:
            affineActiveListKernelImpl1B_N5(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 6:
            affineActiveListKernelImpl1B_N6(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 7:
            affineActiveListKernelImpl1B_N7(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    case 8:
            affineActiveListKernelImpl1B_N8(config, al, input, in_ptr,
                numberOfElementsPerGroup, numberOfIterationsPerGroup, vectorTailLength, simdVectorLength);
            break;
    }
}

void initializeVectors(ExecutionKernelConfig<AffineConfig> const * const config,
        int16_t const * input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX], uint32_t simdVectorLength)
{
    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    uint32_t i;

    if (config->RequestConfig->Transform.inputVectorCount == 8)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d7[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 7];
        }
        input[7] = config->Intermediate->d7 + simdVectorLength;
        in_ptr[7] = (__m128i*)config->Intermediate->d7;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 7)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d6[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 6];
        }
        input[6] = config->Intermediate->d6 + simdVectorLength;
        in_ptr[6] = (__m128i*)config->Intermediate->d6;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 6)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d5[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 5];
        }
        input[5] = config->Intermediate->d5 + simdVectorLength;
        in_ptr[5] = (__m128i*)config->Intermediate->d5;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 5)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d4[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 4];
        }
        input[4] = config->Intermediate->d4 + simdVectorLength;
        in_ptr[4] = (__m128i*)config->Intermediate->d4;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 4)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d3[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 3];
        }
        input[3] = config->Intermediate->d3 + simdVectorLength;
        in_ptr[3] = (__m128i*)config->Intermediate->d3;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 3)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d2[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 2];
        }
        input[2] = config->Intermediate->d2 + simdVectorLength;
        in_ptr[2] = (__m128i*)config->Intermediate->d2;
    }
    if (config->RequestConfig->Transform.inputVectorCount >= 2)
    {
        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d1[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount + 1];
        }
        input[1] = config->Intermediate->d1 + simdVectorLength;
        in_ptr[1] = (__m128i*)config->Intermediate->d1;

        for (i = 0; i < config->RequestConfig->Transform.inputElementCount; i++)
        {
            config->Intermediate->d0[i] = inputs[i*config->RequestConfig->Transform.inputVectorCount];
        }
        input[0] = config->Intermediate->d0 + simdVectorLength;
        in_ptr[0] = (__m128i*)config->Intermediate->d0;
    }
}

void affineActiveListKernelImpl1B_N1(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int16_t const *inputs = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    // simd weights
    __m128i w;
    __m128i w0;
    __m128i w1;

    // simd inputs
    __m128i in0;
    __m128i in1;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;

    // simd accumulators' sums
    int64_t sum0;

    in_ptr[0] = (__m128i*)config->RequestConfig->Inputs;
    input[0] = inputs + simdVectorLength;
    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;

        ix = 0;
        acc0 = _mm_setzero_si128();
        acc1 = _mm_setzero_si128();
        sum0 = bias->bias;

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            acc0 = _mm_add_epi32(acc0, acc1);
            sum0 += vec_sum32(acc0) * bias->multiplier;
            saturate_store_out(&sum0, output, config->SaturationCount);
            sum0 = *output;

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

        for (j = 0; j < vectorTailLength; j++, weight++)
        {
            sum0 += (*input)[j] * *weight++ * bias->multiplier;
        }

        saturate_store_out(&sum0, output, config->SaturationCount);

        output++;
    }

}

void affineActiveListKernelImpl1B_N2(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    // simd weights
    __m128i w;

    // simd inputs
    __m128i in0;
    __m128i in1;

    // simd accumulators
    __m128i acc0;
    __m128i acc1;

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
        ix = 0;

        sum0 = bias->bias;
        sum1 = bias->bias;

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

void affineActiveListKernelImpl1B_N3(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

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

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
        ix = 0;

        sum0 = bias->bias;
        sum1 = bias->bias;
        sum2 = bias->bias;

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);
            saturate(&sum2, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

void affineActiveListKernelImpl1B_N4(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

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

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
        ix = 0;

        sum0 = bias->bias;
        sum1 = bias->bias;
        sum2 = bias->bias;
        sum3 = bias->bias;

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);
            saturate(&sum2, config->SaturationCount);
            saturate(&sum3, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

void affineActiveListKernelImpl1B_N5(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfSimdIterations;
    uint32_t remainderOfSimdIterations;
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

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

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
        ix = 0;

        sum0 = bias->bias;
        sum1 = bias->bias;
        sum2 = bias->bias;
        sum3 = bias->bias;
        sum4 = bias->bias;

        for (kk = 0; kk < numberOfIterationsPerGroup + 1; kk++)
        {
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);
            saturate(&sum2, config->SaturationCount);
            saturate(&sum3, config->SaturationCount);
            saturate(&sum4, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

void affineActiveListKernelImpl1B_N6(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

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

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
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
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);
            saturate(&sum2, config->SaturationCount);
            saturate(&sum3, config->SaturationCount);
            saturate(&sum4, config->SaturationCount);
            saturate(&sum5, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->SaturationCount);
        saturate_store_out(&sum5, &output[5], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

void affineActiveListKernelImpl1B_N7(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

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

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;
    int64_t sum6;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
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
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);
            saturate(&sum2, config->SaturationCount);
            saturate(&sum3, config->SaturationCount);
            saturate(&sum4, config->SaturationCount);
            saturate(&sum5, config->SaturationCount);
            saturate(&sum6, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->SaturationCount);
        saturate_store_out(&sum5, &output[5], config->SaturationCount);
        saturate_store_out(&sum6, &output[6], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

void affineActiveListKernelImpl1B_N8(
    ExecutionKernelConfig<AffineConfig> const * config, AffineConfigAl al, int16_t const *input[XNN_N_GROUP_MAX], __m128i *in_ptr[XNN_N_GROUP_MAX],
    uint32_t numberOfElementsPerGroup, uint32_t numberOfIterationsPerGroup, uint32_t vectorTailLength, uint32_t simdVectorLength)
{
    uint32_t numberOfIterations;
    uint32_t ix_end;
    uint32_t ix;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t l;

    int8_t const * weight;
    nn_bias_c const * bias;
    int32_t * output;

    output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

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

    // simd accumulators' sums
    int64_t sum0;
    int64_t sum1;
    int64_t sum2;
    int64_t sum3;
    int64_t sum4;
    int64_t sum5;
    int64_t sum6;
    int64_t sum7;

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        weight = config->RequestConfig->Transform.weights1B+i*config->RequestConfig->Transform.inputElementCount;
        bias = config->RequestConfig->Transform.biasesCompound+i;
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
            saturate(&sum0, config->SaturationCount);
            saturate(&sum1, config->SaturationCount);
            saturate(&sum2, config->SaturationCount);
            saturate(&sum3, config->SaturationCount);
            saturate(&sum4, config->SaturationCount);
            saturate(&sum5, config->SaturationCount);
            saturate(&sum6, config->SaturationCount);
            saturate(&sum7, config->SaturationCount);

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

        saturate_store_out(&sum0, &output[0], config->SaturationCount);
        saturate_store_out(&sum1, &output[1], config->SaturationCount);
        saturate_store_out(&sum2, &output[2], config->SaturationCount);
        saturate_store_out(&sum3, &output[3], config->SaturationCount);
        saturate_store_out(&sum4, &output[4], config->SaturationCount);
        saturate_store_out(&sum5, &output[5], config->SaturationCount);
        saturate_store_out(&sum6, &output[6], config->SaturationCount);
        saturate_store_out(&sum7, &output[7], config->SaturationCount);

        output += config->RequestConfig->Transform.inputVectorCount;
    }

}

