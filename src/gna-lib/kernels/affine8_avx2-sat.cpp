/*
 INTEL CONFIDENTIAL
 Copyright 2021 Intel Corporation.

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
#include "common_avx2.hpp"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

static void AffineKernelImpl1B1B_N1(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N2(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N3(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N4(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N5(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N6(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N7(ExecutionKernelConfig<AffineConfig> const *const config);
static void AffineKernelImpl1B1B_N8(ExecutionKernelConfig<AffineConfig> const *const config);

/** @brief Affine kernel implementation for 1B input 1B weight
 *
 * ASSUMPTIONS:
 *   Input is in KxN where K [16, 2^16 - 16] mod 16, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 *
 * Under these asumptions most of the operations won't saturate. Only adding 4b bias can saturate.
 */
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 1:
        AffineKernelImpl1B1B_N1(config);
        break;
    case 2:
        AffineKernelImpl1B1B_N2(config);
        break;
    case 3:
        AffineKernelImpl1B1B_N3(config);
        break;
    case 4:
        AffineKernelImpl1B1B_N4(config);
        break;
    case 5:
        AffineKernelImpl1B1B_N5(config);
        break;
    case 6:
        AffineKernelImpl1B1B_N6(config);
        break;
    case 7:
        AffineKernelImpl1B1B_N7(config);
        break;
    case 8:
        AffineKernelImpl1B1B_N8(config);
        break;
    default:
        break;
    }
}

void AffineKernelImpl1B1B_N1(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 1;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;

    __m256i input0_16;

    __m256i sum0;

    __m256i mul0;

    __m256i sum0_lo32;

    __m256i sum0_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            weight16 = _mm256_cvtepi8_epi16(weight);
            input0_16 = _mm256_cvtepi8_epi16(input0);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);

        outputs0 += N;
    }
}

void AffineKernelImpl1B1B_N2(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 2;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;

    __m256i input0_16;
    __m256i input1_16;

    __m256i sum0;
    __m256i sum1;

    __m256i mul0;
    __m256i mul1;

    __m256i sum0_lo32;
    __m256i sum1_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
    }
}

void AffineKernelImpl1B1B_N3(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 3;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;
    int8_t const *inputs2 = inputs1 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;
    int32_t *outputs2 = outputs1 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;
    __m128i input2;

    __m256i input0_16;
    __m256i input1_16;
    __m256i input2_16;

    __m256i sum0;
    __m256i sum1;
    __m256i sum2;

    __m256i mul0;
    __m256i mul1;
    __m256i mul2;

    __m256i sum0_lo32;
    __m256i sum1_lo32;
    __m256i sum2_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;
    __m256i sum2_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();
        sum2 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));
            input2 = _mm_load_si128((__m128i *)(inputs2 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);
            input2_16 = _mm256_cvtepi8_epi16(input2);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);
            mul2 = _mm256_mullo_epi16(input2_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));
            sum2_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul2));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));
            sum2_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul2, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);
            sum2 = _mm256_add_epi32(sum2, sum2_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
            sum2 = _mm256_add_epi32(sum2, sum2_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);
        *outputs2 = _mm256_hsum_epi32(sum2);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);
        saturate_add(outputs2, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
        outputs2 += N;
    }
}

void AffineKernelImpl1B1B_N4(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 4;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;
    int8_t const *inputs2 = inputs1 + K;
    int8_t const *inputs3 = inputs2 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;
    int32_t *outputs2 = outputs1 + 1;
    int32_t *outputs3 = outputs2 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;
    __m128i input2;
    __m128i input3;

    __m256i input0_16;
    __m256i input1_16;
    __m256i input2_16;
    __m256i input3_16;

    __m256i sum0;
    __m256i sum1;
    __m256i sum2;
    __m256i sum3;

    __m256i mul0;
    __m256i mul1;
    __m256i mul2;
    __m256i mul3;

    __m256i sum0_lo32;
    __m256i sum1_lo32;
    __m256i sum2_lo32;
    __m256i sum3_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;
    __m256i sum2_hi32;
    __m256i sum3_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();
        sum2 = _mm256_setzero_si256();
        sum3 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));
            input2 = _mm_load_si128((__m128i *)(inputs2 + inputOffset));
            input3 = _mm_loadu_si128((__m128i *)(inputs3 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);
            input2_16 = _mm256_cvtepi8_epi16(input2);
            input3_16 = _mm256_cvtepi8_epi16(input3);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);
            mul2 = _mm256_mullo_epi16(input2_16, weight16);
            mul3 = _mm256_mullo_epi16(input3_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));
            sum2_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul2));
            sum3_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul3));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));
            sum2_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul2, 1));
            sum3_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul3, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);
            sum2 = _mm256_add_epi32(sum2, sum2_lo32);
            sum3 = _mm256_add_epi32(sum3, sum3_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
            sum2 = _mm256_add_epi32(sum2, sum2_hi32);
            sum3 = _mm256_add_epi32(sum3, sum3_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);
        *outputs2 = _mm256_hsum_epi32(sum2);
        *outputs3 = _mm256_hsum_epi32(sum3);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);
        saturate_add(outputs2, bias, config->SaturationCount);
        saturate_add(outputs3, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
        outputs2 += N;
        outputs3 += N;
    }
}

void AffineKernelImpl1B1B_N5(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 5;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;
    int8_t const *inputs2 = inputs1 + K;
    int8_t const *inputs3 = inputs2 + K;
    int8_t const *inputs4 = inputs3 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;
    int32_t *outputs2 = outputs1 + 1;
    int32_t *outputs3 = outputs2 + 1;
    int32_t *outputs4 = outputs3 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;
    __m128i input2;
    __m128i input3;
    __m128i input4;

    __m256i input0_16;
    __m256i input1_16;
    __m256i input2_16;
    __m256i input3_16;
    __m256i input4_16;

    __m256i sum0;
    __m256i sum1;
    __m256i sum2;
    __m256i sum3;
    __m256i sum4;

    __m256i mul0;
    __m256i mul1;
    __m256i mul2;
    __m256i mul3;
    __m256i mul4;

    __m256i sum0_lo32;
    __m256i sum1_lo32;
    __m256i sum2_lo32;
    __m256i sum3_lo32;
    __m256i sum4_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;
    __m256i sum2_hi32;
    __m256i sum3_hi32;
    __m256i sum4_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();
        sum2 = _mm256_setzero_si256();
        sum3 = _mm256_setzero_si256();
        sum4 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));
            input2 = _mm_load_si128((__m128i *)(inputs2 + inputOffset));
            input3 = _mm_loadu_si128((__m128i *)(inputs3 + inputOffset));
            input4 = _mm_load_si128((__m128i *)(inputs4 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);
            input2_16 = _mm256_cvtepi8_epi16(input2);
            input3_16 = _mm256_cvtepi8_epi16(input3);
            input4_16 = _mm256_cvtepi8_epi16(input4);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);
            mul2 = _mm256_mullo_epi16(input2_16, weight16);
            mul3 = _mm256_mullo_epi16(input3_16, weight16);
            mul4 = _mm256_mullo_epi16(input4_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));
            sum2_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul2));
            sum3_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul3));
            sum4_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul4));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));
            sum2_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul2, 1));
            sum3_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul3, 1));
            sum4_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul4, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);
            sum2 = _mm256_add_epi32(sum2, sum2_lo32);
            sum3 = _mm256_add_epi32(sum3, sum3_lo32);
            sum4 = _mm256_add_epi32(sum4, sum4_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
            sum2 = _mm256_add_epi32(sum2, sum2_hi32);
            sum3 = _mm256_add_epi32(sum3, sum3_hi32);
            sum4 = _mm256_add_epi32(sum4, sum4_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);
        *outputs2 = _mm256_hsum_epi32(sum2);
        *outputs3 = _mm256_hsum_epi32(sum3);
        *outputs4 = _mm256_hsum_epi32(sum4);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);
        saturate_add(outputs2, bias, config->SaturationCount);
        saturate_add(outputs3, bias, config->SaturationCount);
        saturate_add(outputs4, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
        outputs2 += N;
        outputs3 += N;
        outputs4 += N;
    }
}

void AffineKernelImpl1B1B_N6(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 6;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;
    int8_t const *inputs2 = inputs1 + K;
    int8_t const *inputs3 = inputs2 + K;
    int8_t const *inputs4 = inputs3 + K;
    int8_t const *inputs5 = inputs4 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;
    int32_t *outputs2 = outputs1 + 1;
    int32_t *outputs3 = outputs2 + 1;
    int32_t *outputs4 = outputs3 + 1;
    int32_t *outputs5 = outputs4 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;
    __m128i input2;
    __m128i input3;
    __m128i input4;
    __m128i input5;

    __m256i input0_16;
    __m256i input1_16;
    __m256i input2_16;
    __m256i input3_16;
    __m256i input4_16;
    __m256i input5_16;

    __m256i sum0;
    __m256i sum1;
    __m256i sum2;
    __m256i sum3;
    __m256i sum4;
    __m256i sum5;

    __m256i mul0;
    __m256i mul1;
    __m256i mul2;
    __m256i mul3;
    __m256i mul4;
    __m256i mul5;

    __m256i sum0_lo32;
    __m256i sum1_lo32;
    __m256i sum2_lo32;
    __m256i sum3_lo32;
    __m256i sum4_lo32;
    __m256i sum5_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;
    __m256i sum2_hi32;
    __m256i sum3_hi32;
    __m256i sum4_hi32;
    __m256i sum5_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();
        sum2 = _mm256_setzero_si256();
        sum3 = _mm256_setzero_si256();
        sum4 = _mm256_setzero_si256();
        sum5 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));
            input2 = _mm_load_si128((__m128i *)(inputs2 + inputOffset));
            input3 = _mm_loadu_si128((__m128i *)(inputs3 + inputOffset));
            input4 = _mm_load_si128((__m128i *)(inputs4 + inputOffset));
            input5 = _mm_loadu_si128((__m128i *)(inputs5 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);
            input2_16 = _mm256_cvtepi8_epi16(input2);
            input3_16 = _mm256_cvtepi8_epi16(input3);
            input4_16 = _mm256_cvtepi8_epi16(input4);
            input5_16 = _mm256_cvtepi8_epi16(input5);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);
            mul2 = _mm256_mullo_epi16(input2_16, weight16);
            mul3 = _mm256_mullo_epi16(input3_16, weight16);
            mul4 = _mm256_mullo_epi16(input4_16, weight16);
            mul5 = _mm256_mullo_epi16(input5_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));
            sum2_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul2));
            sum3_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul3));
            sum4_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul4));
            sum5_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul5));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));
            sum2_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul2, 1));
            sum3_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul3, 1));
            sum4_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul4, 1));
            sum5_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul5, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);
            sum2 = _mm256_add_epi32(sum2, sum2_lo32);
            sum3 = _mm256_add_epi32(sum3, sum3_lo32);
            sum4 = _mm256_add_epi32(sum4, sum4_lo32);
            sum5 = _mm256_add_epi32(sum5, sum5_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
            sum2 = _mm256_add_epi32(sum2, sum2_hi32);
            sum3 = _mm256_add_epi32(sum3, sum3_hi32);
            sum4 = _mm256_add_epi32(sum4, sum4_hi32);
            sum5 = _mm256_add_epi32(sum5, sum5_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);
        *outputs2 = _mm256_hsum_epi32(sum2);
        *outputs3 = _mm256_hsum_epi32(sum3);
        *outputs4 = _mm256_hsum_epi32(sum4);
        *outputs5 = _mm256_hsum_epi32(sum5);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);
        saturate_add(outputs2, bias, config->SaturationCount);
        saturate_add(outputs3, bias, config->SaturationCount);
        saturate_add(outputs4, bias, config->SaturationCount);
        saturate_add(outputs5, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
        outputs2 += N;
        outputs3 += N;
        outputs4 += N;
        outputs5 += N;
    }
}

void AffineKernelImpl1B1B_N7(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 7;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;
    int8_t const *inputs2 = inputs1 + K;
    int8_t const *inputs3 = inputs2 + K;
    int8_t const *inputs4 = inputs3 + K;
    int8_t const *inputs5 = inputs4 + K;
    int8_t const *inputs6 = inputs5 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;
    int32_t *outputs2 = outputs1 + 1;
    int32_t *outputs3 = outputs2 + 1;
    int32_t *outputs4 = outputs3 + 1;
    int32_t *outputs5 = outputs4 + 1;
    int32_t *outputs6 = outputs5 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;
    __m128i input2;
    __m128i input3;
    __m128i input4;
    __m128i input5;
    __m128i input6;

    __m256i input0_16;
    __m256i input1_16;
    __m256i input2_16;
    __m256i input3_16;
    __m256i input4_16;
    __m256i input5_16;
    __m256i input6_16;

    __m256i sum0;
    __m256i sum1;
    __m256i sum2;
    __m256i sum3;
    __m256i sum4;
    __m256i sum5;
    __m256i sum6;

    __m256i mul0;
    __m256i mul1;
    __m256i mul2;
    __m256i mul3;
    __m256i mul4;
    __m256i mul5;
    __m256i mul6;

    __m256i sum0_lo32;
    __m256i sum1_lo32;
    __m256i sum2_lo32;
    __m256i sum3_lo32;
    __m256i sum4_lo32;
    __m256i sum5_lo32;
    __m256i sum6_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;
    __m256i sum2_hi32;
    __m256i sum3_hi32;
    __m256i sum4_hi32;
    __m256i sum5_hi32;
    __m256i sum6_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();
        sum2 = _mm256_setzero_si256();
        sum3 = _mm256_setzero_si256();
        sum4 = _mm256_setzero_si256();
        sum5 = _mm256_setzero_si256();
        sum6 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));
            input2 = _mm_load_si128((__m128i *)(inputs2 + inputOffset));
            input3 = _mm_loadu_si128((__m128i *)(inputs3 + inputOffset));
            input4 = _mm_load_si128((__m128i *)(inputs4 + inputOffset));
            input5 = _mm_loadu_si128((__m128i *)(inputs5 + inputOffset));
            input6 = _mm_load_si128((__m128i *)(inputs6 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);
            input2_16 = _mm256_cvtepi8_epi16(input2);
            input3_16 = _mm256_cvtepi8_epi16(input3);
            input4_16 = _mm256_cvtepi8_epi16(input4);
            input5_16 = _mm256_cvtepi8_epi16(input5);
            input6_16 = _mm256_cvtepi8_epi16(input6);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);
            mul2 = _mm256_mullo_epi16(input2_16, weight16);
            mul3 = _mm256_mullo_epi16(input3_16, weight16);
            mul4 = _mm256_mullo_epi16(input4_16, weight16);
            mul5 = _mm256_mullo_epi16(input5_16, weight16);
            mul6 = _mm256_mullo_epi16(input6_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));
            sum2_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul2));
            sum3_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul3));
            sum4_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul4));
            sum5_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul5));
            sum6_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul6));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));
            sum2_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul2, 1));
            sum3_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul3, 1));
            sum4_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul4, 1));
            sum5_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul5, 1));
            sum6_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul6, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);
            sum2 = _mm256_add_epi32(sum2, sum2_lo32);
            sum3 = _mm256_add_epi32(sum3, sum3_lo32);
            sum4 = _mm256_add_epi32(sum4, sum4_lo32);
            sum5 = _mm256_add_epi32(sum5, sum5_lo32);
            sum6 = _mm256_add_epi32(sum6, sum6_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
            sum2 = _mm256_add_epi32(sum2, sum2_hi32);
            sum3 = _mm256_add_epi32(sum3, sum3_hi32);
            sum4 = _mm256_add_epi32(sum4, sum4_hi32);
            sum5 = _mm256_add_epi32(sum5, sum5_hi32);
            sum6 = _mm256_add_epi32(sum6, sum6_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);
        *outputs2 = _mm256_hsum_epi32(sum2);
        *outputs3 = _mm256_hsum_epi32(sum3);
        *outputs4 = _mm256_hsum_epi32(sum4);
        *outputs5 = _mm256_hsum_epi32(sum5);
        *outputs6 = _mm256_hsum_epi32(sum6);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);
        saturate_add(outputs2, bias, config->SaturationCount);
        saturate_add(outputs3, bias, config->SaturationCount);
        saturate_add(outputs4, bias, config->SaturationCount);
        saturate_add(outputs5, bias, config->SaturationCount);
        saturate_add(outputs6, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
        outputs2 += N;
        outputs3 += N;
        outputs4 += N;
        outputs5 += N;
        outputs6 += N;
    }
}

void AffineKernelImpl1B1B_N8(ExecutionKernelConfig<AffineConfig> const *const config)
{
    static const uint32_t IT_STEP = 16;
    static const uint32_t N = 8;
    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *inputs1 = inputs0 + K;
    int8_t const *inputs2 = inputs1 + K;
    int8_t const *inputs3 = inputs2 + K;
    int8_t const *inputs4 = inputs3 + K;
    int8_t const *inputs5 = inputs4 + K;
    int8_t const *inputs6 = inputs5 + K;
    int8_t const *inputs7 = inputs6 + K;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;
    int32_t *outputs1 = outputs0 + 1;
    int32_t *outputs2 = outputs1 + 1;
    int32_t *outputs3 = outputs2 + 1;
    int32_t *outputs4 = outputs3 + 1;
    int32_t *outputs5 = outputs4 + 1;
    int32_t *outputs6 = outputs5 + 1;
    int32_t *outputs7 = outputs6 + 1;

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input0;
    __m128i input1;
    __m128i input2;
    __m128i input3;
    __m128i input4;
    __m128i input5;
    __m128i input6;
    __m128i input7;

    __m256i input0_16;
    __m256i input1_16;
    __m256i input2_16;
    __m256i input3_16;
    __m256i input4_16;
    __m256i input5_16;
    __m256i input6_16;
    __m256i input7_16;

    __m256i sum0;
    __m256i sum1;
    __m256i sum2;
    __m256i sum3;
    __m256i sum4;
    __m256i sum5;
    __m256i sum6;
    __m256i sum7;

    __m256i mul0;
    __m256i mul1;
    __m256i mul2;
    __m256i mul3;
    __m256i mul4;
    __m256i mul5;
    __m256i mul6;
    __m256i mul7;

    __m256i sum0_lo32;
    __m256i sum1_lo32;
    __m256i sum2_lo32;
    __m256i sum3_lo32;
    __m256i sum4_lo32;
    __m256i sum5_lo32;
    __m256i sum6_lo32;
    __m256i sum7_lo32;

    __m256i sum0_hi32;
    __m256i sum1_hi32;
    __m256i sum2_hi32;
    __m256i sum3_hi32;
    __m256i sum4_hi32;
    __m256i sum5_hi32;
    __m256i sum6_hi32;
    __m256i sum7_hi32;

    for (uint32_t i = 0; i < M; ++i)
    {
        sum0 = _mm256_setzero_si256();
        sum1 = _mm256_setzero_si256();
        sum2 = _mm256_setzero_si256();
        sum3 = _mm256_setzero_si256();
        sum4 = _mm256_setzero_si256();
        sum5 = _mm256_setzero_si256();
        sum6 = _mm256_setzero_si256();
        sum7 = _mm256_setzero_si256();

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));

            input0 = _mm_load_si128((__m128i *)(inputs0 + inputOffset));
            input1 = _mm_loadu_si128((__m128i *)(inputs1 + inputOffset));
            input2 = _mm_load_si128((__m128i *)(inputs2 + inputOffset));
            input3 = _mm_loadu_si128((__m128i *)(inputs3 + inputOffset));
            input4 = _mm_load_si128((__m128i *)(inputs4 + inputOffset));
            input5 = _mm_loadu_si128((__m128i *)(inputs5 + inputOffset));
            input6 = _mm_load_si128((__m128i *)(inputs6 + inputOffset));
            input7 = _mm_loadu_si128((__m128i *)(inputs7 + inputOffset));

            weight16 = _mm256_cvtepi8_epi16(weight);

            input0_16 = _mm256_cvtepi8_epi16(input0);
            input1_16 = _mm256_cvtepi8_epi16(input1);
            input2_16 = _mm256_cvtepi8_epi16(input2);
            input3_16 = _mm256_cvtepi8_epi16(input3);
            input4_16 = _mm256_cvtepi8_epi16(input4);
            input5_16 = _mm256_cvtepi8_epi16(input5);
            input6_16 = _mm256_cvtepi8_epi16(input6);
            input7_16 = _mm256_cvtepi8_epi16(input7);

            // NOTE: since weights and inputs0 are 1b, multiplication will not overflow 2b
            mul0 = _mm256_mullo_epi16(input0_16, weight16);
            mul1 = _mm256_mullo_epi16(input1_16, weight16);
            mul2 = _mm256_mullo_epi16(input2_16, weight16);
            mul3 = _mm256_mullo_epi16(input3_16, weight16);
            mul4 = _mm256_mullo_epi16(input4_16, weight16);
            mul5 = _mm256_mullo_epi16(input5_16, weight16);
            mul6 = _mm256_mullo_epi16(input6_16, weight16);
            mul7 = _mm256_mullo_epi16(input7_16, weight16);

            sum0_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul0));
            sum1_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul1));
            sum2_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul2));
            sum3_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul3));
            sum4_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul4));
            sum5_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul5));
            sum6_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul6));
            sum7_lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul7));

            sum0_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul0, 1));
            sum1_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul1, 1));
            sum2_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul2, 1));
            sum3_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul3, 1));
            sum4_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul4, 1));
            sum5_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul5, 1));
            sum6_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul6, 1));
            sum7_hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul7, 1));

            sum0 = _mm256_add_epi32(sum0, sum0_lo32);
            sum1 = _mm256_add_epi32(sum1, sum1_lo32);
            sum2 = _mm256_add_epi32(sum2, sum2_lo32);
            sum3 = _mm256_add_epi32(sum3, sum3_lo32);
            sum4 = _mm256_add_epi32(sum4, sum4_lo32);
            sum5 = _mm256_add_epi32(sum5, sum5_lo32);
            sum6 = _mm256_add_epi32(sum6, sum6_lo32);
            sum7 = _mm256_add_epi32(sum7, sum7_lo32);

            sum0 = _mm256_add_epi32(sum0, sum0_hi32);
            sum1 = _mm256_add_epi32(sum1, sum1_hi32);
            sum2 = _mm256_add_epi32(sum2, sum2_hi32);
            sum3 = _mm256_add_epi32(sum3, sum3_hi32);
            sum4 = _mm256_add_epi32(sum4, sum4_hi32);
            sum5 = _mm256_add_epi32(sum5, sum5_hi32);
            sum6 = _mm256_add_epi32(sum6, sum6_hi32);
            sum7 = _mm256_add_epi32(sum7, sum7_hi32);
        }

        *outputs0 = _mm256_hsum_epi32(sum0);
        *outputs1 = _mm256_hsum_epi32(sum1);
        *outputs2 = _mm256_hsum_epi32(sum2);
        *outputs3 = _mm256_hsum_epi32(sum3);
        *outputs4 = _mm256_hsum_epi32(sum4);
        *outputs5 = _mm256_hsum_epi32(sum5);
        *outputs6 = _mm256_hsum_epi32(sum6);
        *outputs7 = _mm256_hsum_epi32(sum7);

        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        // NOTE: This can saturate only because bias can be 4b
        saturate_add(outputs0, bias, config->SaturationCount);
        saturate_add(outputs1, bias, config->SaturationCount);
        saturate_add(outputs2, bias, config->SaturationCount);
        saturate_add(outputs3, bias, config->SaturationCount);
        saturate_add(outputs4, bias, config->SaturationCount);
        saturate_add(outputs5, bias, config->SaturationCount);
        saturate_add(outputs6, bias, config->SaturationCount);
        saturate_add(outputs7, bias, config->SaturationCount);

        outputs0 += N;
        outputs1 += N;
        outputs2 += N;
        outputs3 += N;
        outputs4 += N;
        outputs5 += N;
        outputs6 += N;
        outputs7 += N;
    }
}
