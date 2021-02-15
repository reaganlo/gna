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
#include "igemv16.h"
#include "common_avx2.hpp"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

template <int N>
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const *const config);

/** @brief Specialisation for N = 1 which compilers tend to generate much less efficient than others */
template <>
void AffineKernelImpl2B1B<1>(ExecutionKernelConfig<AffineConfig> const *const config);

/** @brief Affine kernel implementation for 1B input 2B weight
 *
 * ASSUMPTIONS:
 *   Input is in KxN where K [16, 2^16 - 16] mod 16, N [1,8]
 *   Output is MxN where M [1, 2^16]
 *   Biases can be 1b, 2b or 4b
 * 
 * NOTE:
 *   HW compatibility saturation is used which requires that buffer size is divisible by (N*8)
 *
 */
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    switch (config->RequestConfig->Transform.inputVectorCount)
    {
    case 1:
        AffineKernelImpl2B1B<1>(config);
        break;
    case 2:
        AffineKernelImpl2B1B<2>(config);
        break;
    case 3:
        AffineKernelImpl2B1B<3>(config);
        break;
    case 4:
        AffineKernelImpl2B1B<4>(config);
        break;
    case 5:
        AffineKernelImpl2B1B<5>(config);
        break;
    case 6:
        AffineKernelImpl2B1B<6>(config);
        break;
    case 7:
        AffineKernelImpl2B1B<7>(config);
        break;
    case 8:
        AffineKernelImpl2B1B<8>(config);
        break;
    default:
        break;
    }
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

template <int N>
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const *const config)
{
    // NOTE: For 1B data and 2B weight, the 513rd sum can overflow
    static const uint32_t PARTIAL_SUM_LIMIT = 512;

    static const uint32_t IT_STEP = 8;

    // NOTE: For compatibility with HW we limit the amount of unsaturated sums based on the buffer size
    const uint32_t unsaturated_sum_limit = (config->BufferElementCount[N - 1]) / N / IT_STEP;

    uint32_t K = config->RequestConfig->Transform.inputElementCount;
    uint32_t M = config->RequestConfig->Transform.outputElementCount;
    uint32_t KK = K / IT_STEP;

    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int8_t const *inputs[8] = {(int8_t *)config->Intermediate->d0};
    int32_t *outputs[8] = {reinterpret_cast<int32_t *>(config->RequestConfig->Outputs)};

    size_t inputOffset = 0;
    size_t weightOffset = 0;

    __m128i weight;
    __m256i weight32;

    __m128i input[8];
    __m256i input_32[8];
    __m256i sum[8];
    __m256i sum_partial[8];
    __m256i mul[8];

    int64_t final_sum[8];

    uint32_t partial_sum_counter = 0;
    uint32_t unsaturated_sum_counter = 0;

    for (uint32_t n = 1; n < N; ++n)
    {
        inputs[n] = inputs[n - 1] + K;
        outputs[n] = outputs[n - 1] + 1;
    }

    for (uint32_t i = 0; i < M; ++i)
    {
        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

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
                final_sum[n] += _mm256_hsum_epi64(sum[n]);
            }
            partial_sum_counter = 0;
        }

        for (uint32_t n = 0; n < N; ++n)
        {
            saturate_store_out(&final_sum[n], outputs[n], config->SaturationCount);
            outputs[n] += N;
        }
    }
}

template <>
void AffineKernelImpl2B1B<1>(ExecutionKernelConfig<AffineConfig> const *const config)
{
    // NOTE: For 1B data and 2B weight, the 513rd sum can overflow
    static const uint32_t PARTIAL_SUM_LIMIT = 512;

    static const uint32_t IT_STEP = 8;
    static const int N = 1;

    // NOTE: For compatibility with HW we limit the amount of unsaturated sums based on the buffer size
    const uint32_t unsaturated_sum_limit = (config->BufferElementCount[N - 1]) / N / IT_STEP;

    const uint32_t K = config->RequestConfig->Transform.inputElementCount;
    const uint32_t M = config->RequestConfig->Transform.outputElementCount;
    const uint32_t KK = K / IT_STEP;

    int8_t const *inputs0 = (int8_t *)config->Intermediate->d0;
    int8_t const *weights = config->RequestConfig->Transform.weights1B;

    int32_t *outputs = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int32_t *outputs0 = outputs;

    size_t inputOffset = 0;
    size_t weightOffset = 0;

    __m128i weight;
    __m256i weight32;
    __m128i input0;
    __m256i input0_32;
    __m256i sum0;
    __m256i sum0_partial;
    __m256i mul0;

    uint32_t partial_sum_counter = 0;
    uint32_t unsaturated_sum_counter = 0;

    for (uint32_t i = 0; i < M; ++i)
    {
        int32_t bias = getBias(config->RequestConfig->Transform.biasesSimple,
                               config->RequestConfig->Transform.bytesPerBias, i);

        int64_t final_sum0 = bias;

        sum0 = _mm256_setzero_si256();
        sum0_partial = _mm256_setzero_si256();

        unsaturated_sum_counter = 0;

        for (uint32_t k = 0; k < KK; ++k)
        {
            inputOffset = k * IT_STEP;
            weightOffset = (inputOffset + i * K) * sizeof(int16_t);

            input0 = _mm_loadl_epi64((__m128i *)(inputs0 + inputOffset));
            weight = _mm_load_si128((__m128i *)(weights + weightOffset));
            weight32 = _mm256_cvtepi16_epi32(weight);
            input0_32 = _mm256_cvtepi8_epi32(input0);
            mul0 = _mm256_mullo_epi32(input0_32, weight32);
            sum0_partial = _mm256_add_epi32(sum0_partial, mul0);

            if (++partial_sum_counter >= PARTIAL_SUM_LIMIT)
            {
                PartialSum(sum0, sum0_partial);
                partial_sum_counter = 0;
            }

            // NOTE: This part is only for HW compatibility
            if (++unsaturated_sum_counter >= unsaturated_sum_limit)
            {
                if (partial_sum_counter > 0)
                {
                    PartialSum(sum0, sum0_partial);
                    partial_sum_counter = 0;
                }

                final_sum0 += _mm256_hsum_epi64(sum0);

                saturate(&final_sum0, config->SaturationCount);

                sum0 = _mm256_setzero_si256();

                unsaturated_sum_counter = 0;
            }
        }

        if (unsaturated_sum_counter > 0)
        {
            if (partial_sum_counter > 0)
            {
                PartialSum(sum0, sum0_partial);
                partial_sum_counter = 0;
            }

            final_sum0 += _mm256_hsum_epi64(sum0);
        }

        saturate_store_out(&final_sum0, outputs0, config->SaturationCount);

        outputs0 += N;
    }
}
