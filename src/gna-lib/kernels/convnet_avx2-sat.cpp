/*
 INTEL CONFIDENTIAL
 Copyright 2017-2021 Intel Corporation.

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

// TODO: make naming convention consistent with other kernel implementations

#include "convnet.h"
#include "igemv.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <immintrin.h>
#include <cstdint>

void SumPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t *P, int64_t *V)
{
    uint32_t k = 0;
    uint32_t index = 0;
    *V = 0;

    for (k = 0; k < PNE; k++)
    {
        index = (PSI + k) % PS;
        *V += P[index];
    }
}

void MaxPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t *P, int64_t *V)
{
    uint32_t k = 0;
    uint32_t index = 0;
    *V = P[PSI % PS];

    for (k = 0; k < PNE; k++)
    {
        index = (PSI + k) % PS;
        if (P[index]>(*V))
        {
            *V = P[index];
        }
    }
}

void ConvolutionKernelImpl(ConvolutionConfig const * const filterConfig)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int16_t* const I = filterConfig->inputs;
    const int16_t* const F = filterConfig->filters;
    const auto * const B = reinterpret_cast<uint8_t const *>(filterConfig->biases);
    int32_t * const O = filterConfig->convolutedOutputs;
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    uint32_t i;
    uint32_t j;

    gna_sum_t sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;

    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;

#if OPT_LEVEL > 1
    mm_ptr in1, in2, in3, in4, in5, in6, in7, in8, in_end, flt;
    int32_t *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
    const uint8_t * bias;

#if OPT_LEVEL == 4 || OPT_LEVEL == 5
    __m256i f, v1, v2, v3, v4, v5, v6, v7, v8;
#else
    mm_vector f, v1, v2, v3, v4, v5, v6, v7, v8;
#endif

    mm_vector acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8,
            im1, im2, im3, im4, im5, im6, im7, im8;

    uint32_t FC_REM = FC % VEC_16CAP;
    uint32_t FC_VEC = FC - FC_REM;
    uint8_t gr = 8;
    uint32_t N_REM = num_filter_outputs % gr;
    uint32_t N_VEC = num_filter_outputs - N_REM;

    for (j = 0; j < N_VEC; j += gr)
    {
        in_end = (mm_ptr)(I + j * num_inputs_band_stride + FC_VEC);

        out1 = O + j * FN;
        out2 = out1 + FN;
        out3 = out2 + FN;
        out4 = out3 + FN;
        out5 = out4 + FN;
        out6 = out5 + FN;
        out7 = out6 + FN;
        out8 = out7 + FN;

        for (bias = B, i = 0; i < FN; i++)
        {
            flt = (mm_ptr)(F + i * FC);
            f = vec_lddqu(flt);

            in1 = (mm_ptr)(I + j * num_inputs_band_stride);
            in2 = (mm_ptr)(I + (j + 1) * num_inputs_band_stride);
            in3 = (mm_ptr)(I + (j + 2) * num_inputs_band_stride);
            in4 = (mm_ptr)(I + (j + 3) * num_inputs_band_stride);
            in5 = (mm_ptr)(I + (j + 4) * num_inputs_band_stride);
            in6 = (mm_ptr)(I + (j + 5) * num_inputs_band_stride);
            in7 = (mm_ptr)(I + (j + 6) * num_inputs_band_stride);
            in8 = (mm_ptr)(I + (j + 7) * num_inputs_band_stride);

            v1 = vec_lddqu(in1);
            v2 = vec_lddqu(in2);
            v3 = vec_lddqu(in3);
            v4 = vec_lddqu(in4);
            v5 = vec_lddqu(in5);
            v6 = vec_lddqu(in6);
            v7 = vec_lddqu(in7);
            v8 = vec_lddqu(in8);

            acc1 = vec_setzero();
            acc2 = vec_setzero();
            acc3 = vec_setzero();
            acc4 = vec_setzero();
            acc5 = vec_setzero();
            acc6 = vec_setzero();
            acc7 = vec_setzero();
            acc8 = vec_setzero();

            for (; in1 < in_end; )
            {
                in1++;
                in2++;
                in3++;
                in4++;
                in5++;
                in6++;
                in7++;
                in8++;
                flt++;

                im1 = vec_madd16(v1, f);
                im2 = vec_madd16(v2, f);
                im3 = vec_madd16(v3, f);
                im4 = vec_madd16(v4, f);
                im5 = vec_madd16(v5, f);
                im6 = vec_madd16(v6, f);
                im7 = vec_madd16(v7, f);
                im8 = vec_madd16(v8, f);

                acc1 = vec_accumulate(acc1, im1);
                acc2 = vec_accumulate(acc2, im2);
                acc3 = vec_accumulate(acc3, im3);
                acc4 = vec_accumulate(acc4, im4);
                acc5 = vec_accumulate(acc5, im5);
                acc6 = vec_accumulate(acc6, im6);
                acc7 = vec_accumulate(acc7, im7);
                acc8 = vec_accumulate(acc8, im8);

                v1 = vec_lddqu(in1);
                v2 = vec_lddqu(in2);
                v3 = vec_lddqu(in3);
                v4 = vec_lddqu(in4);
                v5 = vec_lddqu(in5);
                v6 = vec_lddqu(in6);
                v7 = vec_lddqu(in7);
                v8 = vec_lddqu(in8);
                f = vec_lddqu(flt);
            }

            sum1 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc1);
            sum2 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc2);
            sum3 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc3);
            sum4 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc4);
            sum5 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc5);
            sum6 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc6);
            sum7 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc7);
            sum8 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc8);

            bias += filterConfig->bytesPerBias;

// FC is mply by 8, for AVX load there might be a tail of 8
#if OPT_LEVEL != 2 && OPT_LEVEL != 3
            if (FC_VEC < FC)
            {
                __m128i s1 = _mm256_castsi256_si128(v1);
                __m128i s2 = _mm256_castsi256_si128(v2);
                __m128i s3 = _mm256_castsi256_si128(v3);
                __m128i s4 = _mm256_castsi256_si128(v4);
                __m128i s5 = _mm256_castsi256_si128(v5);
                __m128i s6 = _mm256_castsi256_si128(v6);
                __m128i s7 = _mm256_castsi256_si128(v7);
                __m128i s8 = _mm256_castsi256_si128(v8);
                __m128i sf = _mm256_castsi256_si128(f);

                s1 = _mm_madd_epi16(s1, sf);
                s2 = _mm_madd_epi16(s2, sf);
                s3 = _mm_madd_epi16(s3, sf);
                s4 = _mm_madd_epi16(s4, sf);
                s5 = _mm_madd_epi16(s5, sf);
                s6 = _mm_madd_epi16(s6, sf);
                s7 = _mm_madd_epi16(s7, sf);
                s8 = _mm_madd_epi16(s8, sf);

                sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                sum5 += _mm_extract_epi32(s5, 0) + _mm_extract_epi32(s5, 1) + _mm_extract_epi32(s5, 2) + _mm_extract_epi32(s5, 3);
                sum6 += _mm_extract_epi32(s6, 0) + _mm_extract_epi32(s6, 1) + _mm_extract_epi32(s6, 2) + _mm_extract_epi32(s6, 3);
                sum7 += _mm_extract_epi32(s7, 0) + _mm_extract_epi32(s7, 1) + _mm_extract_epi32(s7, 2) + _mm_extract_epi32(s7, 3);
                sum8 += _mm_extract_epi32(s8, 0) + _mm_extract_epi32(s8, 1) + _mm_extract_epi32(s8, 2) + _mm_extract_epi32(s8, 3);
            }
#endif

#if GNA_SAT == 1
            saturate_store_out(&sum1, out1, saturationCount);
            saturate_store_out(&sum2, out2, saturationCount);
            saturate_store_out(&sum3, out3, saturationCount);
            saturate_store_out(&sum4, out4, saturationCount);
            saturate_store_out(&sum5, out5, saturationCount);
            saturate_store_out(&sum6, out6, saturationCount);
            saturate_store_out(&sum7, out7, saturationCount);
            saturate_store_out(&sum8, out8, saturationCount);
#else
            *out1 = sum1;
            *out2 = sum2;
            *out3 = sum3;
            *out4 = sum4;
            *out5 = sum5;
            *out6 = sum6;
            *out7 = sum7;
            *out8 = sum8;
#endif

            out1++;
            out2++;
            out3++;
            out4++;
            out5++;
            out6++;
            out7++;
            out8++;
        }
    }

    for (j = N_VEC; j < num_filter_outputs; j++)
    {
        in_end = (mm_ptr)(I + j * num_inputs_band_stride + FC_VEC);

        out1 = O + j * FN;

        for (bias = B, i = 0; i < FN; i++)
        {
            in1 = (mm_ptr)(I + j * num_inputs_band_stride);
            flt = (mm_ptr)(F + i * FC);

            f = vec_lddqu(flt);
            v1 = vec_lddqu(in1);

            acc1 = vec_setzero();

            for (; in1 < in_end; )
            {
                in1++;
                flt++;

                im1 = vec_madd16(v1, f);
                acc1 = vec_accumulate(acc1, im1);

                f = vec_lddqu(flt);
                v1 = vec_lddqu(in1);
            }

            sum1 = getBias(bias, filterConfig->bytesPerBias) + vec_sum(acc1);
            bias += filterConfig->bytesPerBias;
#if OPT_LEVEL != 2 && OPT_LEVEL != 3
            if (FC_VEC < FC)
            {
                __m128i s1 = _mm256_castsi256_si128(v1);
                __m128i sf = _mm256_castsi256_si128(f);

                s1 = _mm_madd_epi16(s1, sf);
                sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
            }
#endif

#if GNA_SAT == 1
            saturate_store_out(&sum1, out1++, saturationCount);
#else
            *out1++ = sum1;
#endif
        }
    }
#else
    const int16_t* ptr_coef;
    const int16_t* ptr_in;
    for (j = 0; j < num_filter_outputs; j++)
    {
        ptr_in = I + j * num_inputs_band_stride;
        for (i = 0; i < FN; i++)
        {
            ptr_coef = F + i * FC;
            sum = B[i];
            for (k = 0; k < FC; k++)
            {
                sum += ptr_in[k] * ptr_coef[k];
            }
#if GNA_SAT == 1
            saturate_store_out(&sum, &O[j * FN + i], saturationCount);
#else
            O[j * FN + i] = sum;
#endif
        }
    }
#endif
}

void ConvolutionPoolingKernelImpl(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int16_t* const I = filterConfig->inputs;
    const int16_t* const F = filterConfig->filters;
    const BiasRegular * const B = filterConfig->biases;
    int16_t * const O = filterConfig->pooledOutputs;
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    const auto PT = poolConfig->Mode;
    const uint32_t PS = poolConfig->Size;
    const uint32_t PSTEP = poolConfig->Step;
    int64_t * const pool = poolConfig->Buffer;

    if (PS == 0)
    {
        return;
    }

    pwl->KERNEL(InitializeActivationFunctions)();

    void(*func_partial_pooling)(const uint32_t PS, const uint32_t pool_num_entries, const uint32_t pool_start_index, const int64_t *P, int64_t *V);

    if (PT == KernelPoolingModeSum)
    {
        func_partial_pooling = SumPartialPoolingFunction;
    }
    else
    {
        func_partial_pooling = MaxPartialPoolingFunction;
    }

    uint32_t pool_start_index = 0;
    uint32_t pool_end_index = 0;
    int32_t pool_num_entries = 0;
    uint32_t output_index = 0;
    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;
    uint32_t i;
    uint32_t j;
    int64_t value;
    uint32_t inc;
    gna_sum_t sum1, sum2, sum3, sum4, sum5, sum6;

#if OPT_LEVEL > 1
    mm_ptr in1, in2, in3, in4, in5, in6, flt, in_end;
#if OPT_LEVEL == 4 || OPT_LEVEL == 5 // AVX1 load vectors
    __m256i v1, v2, v3, v4, v5, v6, f;
#else
    mm_vector v1, v2, v3, v4, v5, v6, f;
#endif
    mm_vector im1, im2, im3, im4, im5, im6;
    mm_vector acc1, acc2, acc3, acc4, acc5, acc6;
#endif

    output_index = 0;
    pool_start_index = 0;
    pool_end_index = 0;
    pool_num_entries = 0;

    for (j = 0; j < num_filter_outputs; )
    {
        if (j >= output_index * PSTEP)
        {
            inc = (PS - static_cast<uint32_t>(pool_num_entries) < num_filter_outputs - j)
                ? PS - static_cast<uint32_t>(pool_num_entries)
                : num_filter_outputs - j;

#if OPT_LEVEL > 1
            uint32_t FC_VEC = FC - FC % VEC_16CAP;
            in_end = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride + FC_VEC);

            // inc <1, 6>
            if (6 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    in4 = (mm_ptr)const_cast<int16_t*>(I + (j + 3) * num_inputs_band_stride);
                    in5 = (mm_ptr)const_cast<int16_t*>(I + (j + 4) * num_inputs_band_stride);
                    in6 = (mm_ptr)const_cast<int16_t*>(I + (j + 5) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    v4 = vec_lddqu(in4);
                    v5 = vec_lddqu(in5);
                    v6 = vec_lddqu(in6);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();
                    acc4 = vec_setzero();
                    acc5 = vec_setzero();
                    acc6 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];
                    sum4 = B[i];
                    sum5 = B[i];
                    sum6 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        in4++;
                        in5++;
                        in6++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);
                        im4 = vec_madd16(v4, f);
                        im5 = vec_madd16(v5, f);
                        im6 = vec_madd16(v6, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);
                        acc4 = vec_accumulate(acc4, im4);
                        acc5 = vec_accumulate(acc5, im5);
                        acc6 = vec_accumulate(acc6, im6);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        v4 = vec_lddqu(in4);
                        v5 = vec_lddqu(in5);
                        v6 = vec_lddqu(in6);
                        f  = vec_lddqu(flt);
                    }

#if OPT_LEVEL != 2 && OPT_LEVEL != 3
                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, s4, s5, s6, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);
                        s4 = _mm256_castsi256_si128(v4);
                        s5 = _mm256_castsi256_si128(v5);
                        s6 = _mm256_castsi256_si128(v6);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);
                        s4 = _mm_madd_epi16(s4, sf);
                        s5 = _mm_madd_epi16(s5, sf);
                        s6 = _mm_madd_epi16(s6, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                        sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                        sum5 += _mm_extract_epi32(s5, 0) + _mm_extract_epi32(s5, 1) + _mm_extract_epi32(s5, 2) + _mm_extract_epi32(s5, 3);
                        sum6 += _mm_extract_epi32(s6, 0) + _mm_extract_epi32(s6, 1) + _mm_extract_epi32(s6, 2) + _mm_extract_epi32(s6, 3);
                    }
#endif

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);
                    sum4 += vec_sum(acc4);
                    sum5 += vec_sum(acc5);
                    sum6 += vec_sum(acc6);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+3)%PS] = sum4;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+4)%PS] = sum5;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+5)%PS] = sum6;
                    pool_end_index = 0;
                }
            }
            if (5 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    in4 = (mm_ptr)const_cast<int16_t*>(I + (j + 3) * num_inputs_band_stride);
                    in5 = (mm_ptr)const_cast<int16_t*>(I + (j + 4) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    v4 = vec_lddqu(in4);
                    v5 = vec_lddqu(in5);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();
                    acc4 = vec_setzero();
                    acc5 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];
                    sum4 = B[i];
                    sum5 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        in4++;
                        in5++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);
                        im4 = vec_madd16(v4, f);
                        im5 = vec_madd16(v5, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);
                        acc4 = vec_accumulate(acc4, im4);
                        acc5 = vec_accumulate(acc5, im5);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        v4 = vec_lddqu(in4);
                        v5 = vec_lddqu(in5);
                        f  = vec_lddqu(flt);
                    }

#if OPT_LEVEL != 2 && OPT_LEVEL != 3
                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, s4, s5, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);
                        s4 = _mm256_castsi256_si128(v4);
                        s5 = _mm256_castsi256_si128(v5);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);
                        s4 = _mm_madd_epi16(s4, sf);
                        s5 = _mm_madd_epi16(s5, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                        sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                        sum5 += _mm_extract_epi32(s5, 0) + _mm_extract_epi32(s5, 1) + _mm_extract_epi32(s5, 2) + _mm_extract_epi32(s5, 3);
                    }
#endif

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);
                    sum4 += vec_sum(acc4);
                    sum5 += vec_sum(acc5);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+3)%PS] = sum4;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+4)%PS] = sum5;
                }
            }
            if (4 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    in4 = (mm_ptr)const_cast<int16_t*>(I + (j + 3) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    v4 = vec_lddqu(in4);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();
                    acc4 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];
                    sum4 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        in4++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);
                        im4 = vec_madd16(v4, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);
                        acc4 = vec_accumulate(acc4, im4);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        v4 = vec_lddqu(in4);
                        f  = vec_lddqu(flt);
                    }

#if OPT_LEVEL != 2 && OPT_LEVEL != 3
                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, s4, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);
                        s4 = _mm256_castsi256_si128(v4);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);
                        s4 = _mm_madd_epi16(s4, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                        sum4 += _mm_extract_epi32(s4, 0) + _mm_extract_epi32(s4, 1) + _mm_extract_epi32(s4, 2) + _mm_extract_epi32(s4, 3);
                    }
#endif

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);
                    sum4 += vec_sum(acc4);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+3)%PS] = sum4;
                }
            }
            if (3 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    in3 = (mm_ptr)const_cast<int16_t*>(I + (j + 2) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    v3 = vec_lddqu(in3);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();
                    acc3 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];
                    sum3 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        in3++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);
                        im3 = vec_madd16(v3, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);
                        acc3 = vec_accumulate(acc3, im3);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        v3 = vec_lddqu(in3);
                        f  = vec_lddqu(flt);
                    }

#if OPT_LEVEL != 2 && OPT_LEVEL != 3
                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, s3, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);
                        s3 = _mm256_castsi256_si128(v3);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);
                        s3 = _mm_madd_epi16(s3, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                        sum3 += _mm_extract_epi32(s3, 0) + _mm_extract_epi32(s3, 1) + _mm_extract_epi32(s3, 2) + _mm_extract_epi32(s3, 3);
                    }
#endif

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);
                    sum3 += vec_sum(acc3);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+2)%PS] = sum3;
                }
            }
            if (2 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    in2 = (mm_ptr)const_cast<int16_t*>(I + (j + 1) * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    v2 = vec_lddqu(in2);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();
                    acc2 = vec_setzero();

                    sum1 = B[i];
                    sum2 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        in2++;
                        flt++;

                        im1 = vec_madd16(v1, f);
                        im2 = vec_madd16(v2, f);

                        acc1 = vec_accumulate(acc1, im1);
                        acc2 = vec_accumulate(acc2, im2);

                        v1 = vec_lddqu(in1);
                        v2 = vec_lddqu(in2);
                        f  = vec_lddqu(flt);
                    }

#if OPT_LEVEL != 2 && OPT_LEVEL != 3
                    if (FC_VEC < FC)
                    {
                        __m128i s1, s2, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);
                        s2 = _mm256_castsi256_si128(v2);

                        s1 = _mm_madd_epi16(s1, sf);
                        s2 = _mm_madd_epi16(s2, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                        sum2 += _mm_extract_epi32(s2, 0) + _mm_extract_epi32(s2, 1) + _mm_extract_epi32(s2, 2) + _mm_extract_epi32(s2, 3);
                    }
#endif

                    sum1 += vec_sum(acc1);
                    sum2 += vec_sum(acc2);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                    pool[i * CNN_POOL_SIZE_MAX + (pool_end_index+1)%PS] = sum2;
                }
            }
            if (1 == inc)
            {
                for (i = 0; i < FN; i++)
                {
                    in1 = (mm_ptr)const_cast<int16_t*>(I + j * num_inputs_band_stride);
                    flt = (mm_ptr)(F + i * FC);

                    v1 = vec_lddqu(in1);
                    f  = vec_lddqu(flt);

                    acc1 = vec_setzero();

                    sum1 = B[i];

                    for (; in1 < in_end; )
                    {
                        in1++;
                        flt++;

                        im1 = vec_madd16(v1, f);

                        acc1 = vec_accumulate(acc1, im1);

                        v1 = vec_lddqu(in1);
                        f  = vec_lddqu(flt);
                    }

#if OPT_LEVEL != 2 && OPT_LEVEL != 3
                    if (FC_VEC < FC)
                    {
                        __m128i s1, sf;
                        sf = _mm256_castsi256_si128(f);
                        s1 = _mm256_castsi256_si128(v1);

                        s1 = _mm_madd_epi16(s1, sf);

                        sum1 += _mm_extract_epi32(s1, 0) + _mm_extract_epi32(s1, 1) + _mm_extract_epi32(s1, 2) + _mm_extract_epi32(s1, 3);
                    }
#endif

                    sum1 += vec_sum(acc1);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                }
            }
            pool_end_index += inc;
            pool_end_index %= PS;

#else
            for (l = 0; l < inc; l++)
            {
                ptr_in = I + (j + l)*num_inputs_band_stride;

                for (i = 0; i < FN; i++)
                {
                    ptr_coef = F + i * FC;

                    sum = B[i];
                    for (k = 0; k < FC; k++)
                    {
                        sum += ptr_in[k] * ptr_coef[k];
                    }

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum;
                }

                pool_end_index = (pool_end_index + 1) % PS;
            }
#endif

            j += inc;
            pool_num_entries += inc;
            if (static_cast<uint32_t>(pool_num_entries) == PS)
            {
                for (i = 0; i < FN; i++)
                {
                    func_partial_pooling(PS, PS, 0, pool + i * CNN_POOL_SIZE_MAX, &value);
#if GNA_SAT == 1
                    gna_saturate_cast(value, *saturationCount);
#endif
                    pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
                }

                pool_start_index = (pool_start_index + PSTEP) % PS;
                pool_num_entries -= PSTEP;
                if (pool_num_entries < 0)
                {
                    pool_start_index = 0;
                    pool_end_index = 0;
                    pool_num_entries = 0;
                }
                output_index++;
            }
        }
        else
        {
            j++;
        }
    }

    while (pool_num_entries > 0)
    {
        for (i = 0; i < FN; i++)
        {
            func_partial_pooling(PS, static_cast<uint32_t>(pool_num_entries), pool_start_index, pool + i * CNN_POOL_SIZE_MAX, &value);
#if GNA_SAT == 1
            gna_saturate_cast(value, *saturationCount);
#endif
            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}

void Convolution2DKernelImpl1B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    uint32_t inputDepth = config->RequestConfig->Transform.InputDepth;
    uint32_t inputHeight = config->RequestConfig->Transform.InputHeight;
    uint32_t inputWidth = config->RequestConfig->Transform.InputWidth;

    uint32_t numFilters = config->RequestConfig->Transform.NumberOfFilters;
    uint32_t filterHeight = config->RequestConfig->Transform.FilterHeight;
    uint32_t filterWidth = config->RequestConfig->Transform.FilterWidth;
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth);
    uint32_t filterPadding = (Gna2RoundUp(memForFilter, 16) - memForFilter);

    uint32_t padHeight = config->RequestConfig->Transform.ZeroPaddingHeight;
    uint32_t padWidth = config->RequestConfig->Transform.ZeroPaddingWidth;

    uint32_t strideHeight = config->RequestConfig->Transform.StrideHeight;
    uint32_t strideWidth = config->RequestConfig->Transform.StrideWidth;

    auto biasMode = config->RequestConfig->Transform.BiasMode;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t inHeightMax = inputHeight + padHeight - 1;

    const int8_t* const I = (int8_t*)config->RequestConfig->Inputs;
    int32_t* O = (int32_t*)config->RequestConfig->Outputs;
    int8_t* F = (int8_t*)config->RequestConfig->Transform.FilterData;

    auto biasPrecission = config->RequestConfig->Transform.BiasDataMode;
    const void* biasData = config->RequestConfig->Transform.BiasData;

    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    const int8_t mask[64] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    for (uint32_t OD = 0; OD < numFilters; OD++)
    { //Output depth or #filters

        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

        for (uint32_t OW = 0; OW < outWidth; OW++)
        { //Output width
            for (uint32_t OH = 0; OH < outHeight; OH++)
            {    //Output height

                int64_t outVal;// = &O[OH * outWidth * numFilters + OW * numFilters + OD]; //NHWC order
                if (biasMode == KernelBiasModePerFilter)
                {
                    outVal = getBias(biasData, biasPrecission, OD);
                }
                else if (biasMode == KernelBiasModeDisabled)
                {
                    outVal = 0;
                }
                else
                {
                    outVal = getBias(biasData, biasPrecission, numFilters*outWidth*OH + numFilters * OW + OD);
                }

                __m256i acc_0 = _mm256_setzero_si256();
                __m256i acc_1 = _mm256_setzero_si256();
                for (uint32_t h = 0; h < filterHeight; h++) {
                    uint32_t inIdxH = (((OH*strideHeight) + h - padHeight) * (inputDepth*inputWidth));
                    uint32_t fIdxH = inputDepth * filterWidth * h;
                    uint32_t hIdx = OH * strideHeight + h;
                    if (hIdx < padHeight || hIdx > inHeightMax) {
                        continue;
                    }
                    uint32_t fIdxW = 0, inIdxW = 0;
                    if (OW * strideWidth < padWidth) {
                        fIdxW = padWidth - OW * strideWidth;
                    } else {
                        inIdxW = OW * strideWidth - padWidth;
                    }
                    uint32_t span = (std::min)(inputWidth - inIdxW, filterWidth - fIdxW);
                    inIdxW *= inputDepth;
                    fIdxW *= inputDepth;
                    span *= inputDepth;
                    uint32_t bound = fIdxW + span;
                    const __m256i masked = _mm256_loadu_si256((const __m256i *)(mask + 32 - (bound-fIdxW)%32));
                    for (; fIdxW < bound; fIdxW += 32, inIdxW += 32) {
                        /* Thanks to the fact that data is packed, we could iterate over W and Z dimensions in one loop.
                         * This observation improves performance a lot for case when W*Z is big, but one of dims is small.
                         * steps:
                         * read both input and filter by 32 1byte elements
                         * possibly zero elements that are past data bound
                         * extend both input and filter from epi8 to epi16 for multiplication
                         * multiply them (result is guaranteed to fit into 16bits, but it does not matter in current impl)
                         * (elements zeroed previously in input now yield zeros, so would not bother later accumulation)
                         * (use _mm256_madd_epi16() instead of _mm256_mullo_epi16(), so we get one level of horizontal sum for free,
                         *  it also expands data to 32bit, as would be needed in accumulation anyway)
                         * accumulate to two registers to decrease dependency chain between loop steps
                         * horizontally add after loop
                         */
                        __m256i   input_epi8 = _mm256_loadu_si256((const __m256i *)(I + inIdxH + inIdxW));
                        __m256i  filter_epi8 = _mm256_loadu_si256((const __m256i *)(F + fIdxN + fIdxH + fIdxW));
                        if (fIdxW+32 > bound) {
                            input_epi8 = _mm256_and_si256(input_epi8, masked);
                        }
                        __m256i filter_epi16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(filter_epi8));
                        __m256i  input_epi16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128( input_epi8));
                        __m256i filter_epi16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(filter_epi8, 1));
                        __m256i  input_epi16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256( input_epi8, 1));
                        __m256i m_0 = _mm256_madd_epi16(input_epi16_0, filter_epi16_0);
                        __m256i m_1 = _mm256_madd_epi16(input_epi16_1, filter_epi16_1);
                        acc_0 = _mm256_add_epi32(acc_0, m_0);
                        acc_1 = _mm256_add_epi32(acc_1, m_1);
                    }
                }
                acc_0 = _mm256_add_epi32(acc_0, acc_1);
                __m128i sum1 = _mm_add_epi32(_mm256_extracti128_si256(acc_0, 1), _mm256_castsi256_si128(acc_0));
                __m128i sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
                __m128i sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
                outVal += _mm_cvtsi128_si32(sum3);
                gna_saturate_cast(outVal, *config->SaturationCount);
                O[OH * outWidth * numFilters + OW * numFilters + OD] = (int32_t)outVal;
            }
        }
    }
}

void Convolution2DKernelImpl1B2B(ExecutionKernelConfig<ConvolutionConfig2D> const *const config)
{
    uint32_t inputDepth = config->RequestConfig->Transform.InputDepth;
    uint32_t inputHeight = config->RequestConfig->Transform.InputHeight;
    uint32_t inputWidth = config->RequestConfig->Transform.InputWidth;

    uint32_t numFilters = config->RequestConfig->Transform.NumberOfFilters;
    uint32_t filterHeight = config->RequestConfig->Transform.FilterHeight;
    uint32_t filterWidth = config->RequestConfig->Transform.FilterWidth;
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth);
    uint32_t filterPadding = (Gna2RoundUp(memForFilter, 16) - memForFilter);

    uint32_t padHeight = config->RequestConfig->Transform.ZeroPaddingHeight;
    uint32_t padWidth = config->RequestConfig->Transform.ZeroPaddingWidth;

    uint32_t strideHeight = config->RequestConfig->Transform.StrideHeight;
    uint32_t strideWidth = config->RequestConfig->Transform.StrideWidth;

    auto biasMode = config->RequestConfig->Transform.BiasMode;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;

    const int16_t *const I = (int16_t *)config->RequestConfig->Inputs;
    int32_t *O = (int32_t *)config->RequestConfig->Outputs;
    int8_t *F = (int8_t *)config->RequestConfig->Transform.FilterData;

    auto biasPrecission = config->RequestConfig->Transform.BiasDataMode;
    const void *biasData = config->RequestConfig->Transform.BiasData;

    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    const int8_t mask[64] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    for (uint32_t OD = 0; OD < numFilters; OD++) {
        uint32_t fIdxN = OD * (inputDepth * filterWidth * filterHeight + filterPadding);

        for (uint32_t OH = 0; OH < outHeight; OH++) {
            for (uint32_t OW = 0; OW < outWidth; OW++) {

                int64_t outVal;
                if (biasMode == KernelBiasModePerFilter) {
                    outVal = getBias(biasData, biasPrecission, OD);
                }
                else if (biasMode == KernelBiasModeDisabled) {
                    outVal = 0;
                }
                else {
                    outVal = getBias(biasData, biasPrecission, numFilters * outWidth * OH + numFilters * OW + OD);
                }

                uint32_t fIdxH = 0, inIdxH = 0;
                if (OH * strideHeight < padHeight) {
                    fIdxH = padHeight - OH * strideHeight;
                }
                else {
                    inIdxH = OH * strideHeight - padHeight;
                }
                uint32_t boundH = (std::min)(filterHeight, inputHeight + padHeight - OH * strideHeight);
                uint32_t steps = boundH - fIdxH;
                inIdxH *= inputDepth * inputWidth;
                fIdxH *= inputDepth * filterWidth;
                uint32_t fIdxW = 0, inIdxW = 0;
                if (OW * strideWidth < padWidth) {
                    fIdxW = padWidth - OW * strideWidth;
                }
                else {
                    inIdxW = OW * strideWidth - padWidth;
                }
                const uint32_t span = (std::min)(inputWidth - inIdxW, filterWidth - fIdxW);
                const uint32_t stepsPerWxZ = span * inputDepth;
                inIdxW *= inputDepth;
                fIdxW *= inputDepth;
                uint32_t idxI = inIdxH + inIdxW, idxF = fIdxN + fIdxH + fIdxW;
                const uint32_t stepsRounded = Gna2RoundUp(stepsPerWxZ, 32);
                uint32_t stepIH = inputDepth * inputWidth - stepsRounded;
                uint32_t stepFH = inputDepth * filterWidth - stepsRounded;
                __m256i acc_0 = _mm256_setzero_si256();
                __m256i acc_1 = _mm256_setzero_si256();
                const __m256i masked = _mm256_loadu_si256((const __m256i *)(mask + 32 - stepsPerWxZ % 32));
                for (; steps--; idxF += stepFH, idxI += stepIH) {
                    for (uint32_t i = 0; i < stepsPerWxZ; i += 32, idxF += 32, idxI += 32) {
                        __m256i input_epi16_0 = _mm256_loadu_si256((const __m256i *)(I + idxI));
                        __m256i input_epi16_1 = _mm256_loadu_si256((const __m256i *)(I + idxI + 16));
                        __m256i filter_epi8 = _mm256_loadu_si256((const __m256i *)(F + idxF));
                        if (i + 32 > stepsPerWxZ) {
                            filter_epi8 = _mm256_and_si256(filter_epi8, masked);
                        }
                        __m256i filter_epi16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(filter_epi8));
                        __m256i filter_epi16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(filter_epi8, 1));
                        __m256i m_0 = _mm256_madd_epi16(input_epi16_0, filter_epi16_0);
                        __m256i m_1 = _mm256_madd_epi16(input_epi16_1, filter_epi16_1);
                        acc_0 = _mm256_add_epi32(acc_0, m_0);
                        acc_1 = _mm256_add_epi32(acc_1, m_1);
                    }
                }
                acc_0 = _mm256_add_epi32(acc_0, acc_1);
                __m128i sum1 = _mm_add_epi32(_mm256_extracti128_si256(acc_0, 1), _mm256_castsi256_si128(acc_0));
                __m128i sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
                __m128i sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
                outVal += _mm_cvtsi128_si32(sum3);
                gna_saturate_cast(outVal, *config->SaturationCount);
                O[numFilters * outWidth * OH + numFilters * OW + OD] = (int32_t)outVal;
            }
        }
    }
}
