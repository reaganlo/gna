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

/** Transpose input and select template for calculation */
static void TransposeAndRun(ExecutionKernelConfig<AffineConfig> const *const config, void *biases, uint32_t bias_vector = 1);

/** Affine kernel implementation for 1B input 1B weight
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    TransposeAndRun(config, (void *)config->RequestConfig->Transform.biasesSimple);
}

/** Affine kernel implementation for 1B input 1B weight, multibias
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    TransposeAndRun(config, (void *)config->RequestConfig->Transform.multiBias, config->RequestConfig->Transform.multiBiasVectorCount);
}

/** Generic implementation for simple bias and multibias
 *
 * This implementation expects input to be already transposed.
 *
 * @param config Execution config
 * @param biases Pointer to either simple bias or multibias
 * @param bias_vector Index of vector used in multibias. For simple bias it must be set to 1
 */
template <size_t N>
static void Affine1B1B(ExecutionKernelConfig<AffineConfig> const *const config, void *biases, const uint32_t bias_vector)
{
    static const uint32_t IT_STEP = 16;

    const uint32_t K = config->RequestConfig->Transform.inputElementCount;
    const uint32_t M = config->RequestConfig->Transform.outputElementCount;
    const uint32_t KK = K / IT_STEP;

    int8_t const *inputs[N] = {(int8_t *)config->Intermediate->d0};
    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs[N] = {reinterpret_cast<int32_t *>(config->RequestConfig->Outputs)};

    uint32_t inputOffset = 0;
    uint32_t weightOffset = 0;

    __m128i weight;
    __m256i weight16;

    __m128i input[N];
    __m256i input_16[N];
    __m256i sum[N];
    __m256i mul[N];
    __m256i sum_lo32[N];
    __m256i sum_hi32[N];

    for (uint32_t n = 1; n < N; ++n)
    {
        inputs[n] = inputs[n - 1] + K;
        outputs[n] = outputs[n - 1] + 1;
    }

    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t n = 0; n < N; ++n)
        {
            sum[n] = _mm256_setzero_si256();
        }

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = inputOffset + i * K;

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));
            weight16 = _mm256_cvtepi8_epi16(weight);

            for (uint32_t n = 0; n < N; ++n)
            {
                input[n] = _mm_load_si128((__m128i *)(inputs[n] + inputOffset));
                input_16[n] = _mm256_cvtepi8_epi16(input[n]);

                // NOTE: since weights and inputs are 1b, multiplication will not overflow 2b
                mul[n] = _mm256_mullo_epi16(input_16[n], weight16);
                sum_lo32[n] = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mul[n]));
                sum_hi32[n] = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mul[n], 1));
                sum[n] = _mm256_add_epi32(sum[n], sum_lo32[n]);
                sum[n] = _mm256_add_epi32(sum[n], sum_hi32[n]);
            }
        }

        const int32_t bias = getBias(biases, config->RequestConfig->Transform.bytesPerBias, i * bias_vector);

        for (uint32_t n = 0; n < N; ++n)
        {
            *outputs[n] = _mm256_hsum_epi32(sum[n]);

            // NOTE: This can saturate only because bias can be 4b
            saturate_add(outputs[n], bias, config->SaturationCount);

            outputs[n] += N;
        }
    }
}

void TransposeAndRun(ExecutionKernelConfig<AffineConfig> const *const config, void *biases, uint32_t bias_vector)
{
    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 1:
        Affine1B1B<1>(config, biases, bias_vector);
        break;
    case 2:
        Affine1B1B<2>(config, biases, bias_vector);
        break;
    case 3:
        Affine1B1B<3>(config, biases, bias_vector);
        break;
    case 4:
        Affine1B1B<4>(config, biases, bias_vector);
        break;
    case 5:
        Affine1B1B<5>(config, biases, bias_vector);
        break;
    case 6:
        Affine1B1B<6>(config, biases, bias_vector);
        break;
    case 7:
        Affine1B1B<7>(config, biases, bias_vector);
        break;
    case 8:
        Affine1B1B<8>(config, biases, bias_vector);
        break;
    default:
        break;
    }
}
