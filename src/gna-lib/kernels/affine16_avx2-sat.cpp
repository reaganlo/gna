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
#include "saturate.h"
#include "igemv8.h"
#include "igemv16.h"
#include "common.hpp"
#include "common_avx2.hpp"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

/** Transpose input and select template for calculation */
template <typename T>
static void TransposeAndRun(ExecutionKernelConfig<AffineConfig> const *const config, T &idx, void *biases, uint32_t bias_vector = 1);

/** Affine kernel implementation for 1B input 2B weight
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    SequenceIndexSource idx(config->RequestConfig->Transform.outputElementCount);
    void *simple_bias = (void *)config->RequestConfig->Transform.biasesSimple;

    TransposeAndRun(config, idx, simple_bias);
}

/** Affine kernel implementation for 1B input 2B weight, multibias
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    SequenceIndexSource idx(config->RequestConfig->Transform.outputElementCount);
    void *multi_bias = (void *)config->RequestConfig->Transform.multiBias;
    uint32_t bias_vector = config->RequestConfig->Transform.multiBiasVectorCount;

    TransposeAndRun(config, idx, multi_bias, bias_vector);
}

/** Affine kernel implementation for 1B input 2B weight, active list
 *
 * ASSUMPTIONS:
 *   Input is KxN where K [16, 2^16 - 16], K % 16 == 0, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 */
void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const *const config, AffineConfigAl al)
{
    void *simple_bias = (void *)config->RequestConfig->Transform.biasesSimple;
    ActiveListIndexSource idx(al);

    TransposeAndRun(config, idx, simple_bias);
}

/** @brief Add 8x32b partial sum to 4x64 total sum
 *
 *  Partial sum is set to zero
 *
 *  @param[in,out] sum 4x64 sum
 *  @param[in,out] partial 8x32 partial sum
 */
inline void PartialSum(__m256i &sum, __m256i &partial)
{
    __m128i partial_lo32 = _mm256_castsi256_si128(partial);
    __m128i partial_hi32 = _mm256_extracti128_si256(partial, 1);

    __m256i partial_lo = _mm256_cvtepi32_epi64(partial_lo32);
    __m256i partial_hi = _mm256_cvtepi32_epi64(partial_hi32);

    sum = _mm256_add_epi64(sum, partial_lo);
    sum = _mm256_add_epi64(sum, partial_hi);

    partial = _mm256_setzero_si256();
}

/** Generic implementation for simple bias, multibias and active list
 *
 * This implementation expects input to be already transposed.
 *
 * @param config Execution config
 * @param idx Index iterator. Must derive from @ref IndexSource. Use @ref SequenceIndexSource for simple bias and multibias and @ref ActiveListIndexSource for Active List
 * @param biases Pointer to either simple bias or multibias
 * @param bias_vector Index of vector used in multibias. For simple bias it must be set to 1
 */
template <size_t N, class T>
static void Affine2B1B(ExecutionKernelConfig<AffineConfig> const *const config, T &idx, void *biases, const uint32_t bias_vector)
{
    static_assert(std::is_base_of<IndexSource, T>::value, "Index iterator must derive from IndexSource");

    // NOTE: For 1B data and 2B weight, the 513rd sum can overflow
    static const uint32_t PARTIAL_SUM_LIMIT = 512;

    static const uint32_t IT_STEP = 8;

    // NOTE: For compatibility with HW we limit the amount of unsaturated sums based on the buffer size
    const uint32_t unsaturated_sum_limit = (uint32_t)((config->BufferElementCount[N - 1]) / N / IT_STEP);

    const uint32_t K = config->RequestConfig->Transform.inputElementCount;
    const uint32_t KK = K / IT_STEP;

    int8_t const *const weights = config->RequestConfig->Transform.weights1B;

    int8_t const *inputs[N] = {(const int8_t *)config->Intermediate->d0};
    int32_t *outputs[N] = {reinterpret_cast<int32_t *>(config->RequestConfig->Outputs)};

    size_t inputOffset = 0;
    size_t weightOffset = 0;

    __m128i weight;
    __m256i weight32;

    __m128i input[N];
    __m256i input_32[N];
    __m256i sum[N];
    __m256i sum_partial[N];
    __m256i mul[N];

    int64_t final_sum[N];

    uint32_t partial_sum_counter = 0;
    uint32_t unsaturated_sum_counter = 0;

    for (uint32_t n = 1; n < N; ++n)
    {
        inputs[n] = inputs[n - 1] + K;
        outputs[n] = outputs[n - 1] + 1;
    }

    while (idx.HasNext())
    {
        uint32_t i = idx.Next();

        const int32_t bias = getBias(biases, config->RequestConfig->Transform.bytesPerBias, i * bias_vector);

        for (uint32_t n = 0; n < N; ++n)
        {
            sum[n] = _mm256_setzero_si256();
            sum_partial[n] = _mm256_setzero_si256();
            final_sum[n] = bias;
        }

        unsaturated_sum_counter = 0;

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = (inputOffset + i * K) * sizeof(int16_t);

            weight = _mm_load_si128((__m128i *)(weights + weightOffset));
            weight32 = _mm256_cvtepi16_epi32(weight);

            for (uint32_t n = 0; n < N; ++n)
            {
                input[n] = _mm_loadl_epi64((__m128i *)(inputs[n] + inputOffset));
                input_32[n] = _mm256_cvtepi8_epi32(input[n]);
                mul[n] = _mm256_mullo_epi32(input_32[n], weight32);
                sum_partial[n] = _mm256_add_epi32(sum_partial[n], mul[n]);
            }

            if (++partial_sum_counter >= PARTIAL_SUM_LIMIT)
            {
                for (uint32_t n = 0; n < N; ++n)
                {
                    PartialSum(sum[n], sum_partial[n]);
                }

                partial_sum_counter = 0;
            }

            // NOTE: This part is only for HW compatibility
            if (++unsaturated_sum_counter >= unsaturated_sum_limit)
            {
                if (partial_sum_counter > 0)
                {
                    for (uint32_t n = 0; n < N; ++n)
                    {
                        PartialSum(sum[n], sum_partial[n]);
                    }

                    partial_sum_counter = 0;
                }

                for (uint32_t n = 0; n < N; ++n)
                {
                    final_sum[n] += _mm256_hsum_epi64(sum[n]);
                    saturate(&final_sum[n], config->SaturationCount);
                    sum[n] = _mm256_setzero_si256();
                }

                unsaturated_sum_counter = 0;
            }
        }

        if (partial_sum_counter > 0)
        {
            for (uint32_t n = 0; n < N; ++n)
            {
                PartialSum(sum[n], sum_partial[n]);
            }
            partial_sum_counter = 0;
        }

        for (uint32_t n = 0; n < N; ++n)
        {
            final_sum[n] += _mm256_hsum_epi64(sum[n]);
            saturate_store_out(&final_sum[n], outputs[n], config->SaturationCount);
            outputs[n] += N;
        }
    }
}

template <typename T>
void TransposeAndRun(ExecutionKernelConfig<AffineConfig> const *const config, T &idx, void *biases, uint32_t bias_vector)
{
    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 1:
        Affine2B1B<1>(config, idx, biases, bias_vector);
        break;
    case 2:
        Affine2B1B<2>(config, idx, biases, bias_vector);
        break;
    case 3:
        Affine2B1B<3>(config, idx, biases, bias_vector);
        break;
    case 4:
        Affine2B1B<4>(config, idx, biases, bias_vector);
        break;
    case 5:
        Affine2B1B<5>(config, idx, biases, bias_vector);
        break;
    case 6:
        Affine2B1B<6>(config, idx, biases, bias_vector);
        break;
    case 7:
        Affine2B1B<7>(config, idx, biases, bias_vector);
        break;
    case 8:
        Affine2B1B<8>(config, idx, biases, bias_vector);
        break;
    default:
        break;
    }
}
