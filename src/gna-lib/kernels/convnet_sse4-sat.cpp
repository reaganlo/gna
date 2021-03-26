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

#include "common_sse4.hpp"
#include "convnet.h"
#include "saturate.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include <cmath>

void SumPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t *P, int64_t* V)
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

void MaxPartialPoolingFunction(const uint32_t PS, const uint32_t PNE, const uint32_t PSI, const int64_t *P, int64_t* V)
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

    mm_ptr in1, in2, in3, in4, in5, in6, in7, in8, in_end, flt;
    int32_t *out1, *out2, *out3, *out4, *out5, *out6, *out7, *out8;
    const uint8_t *bias;

    mm_vector f, v1, v2, v3, v4, v5, v6, v7, v8;

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

            saturate_store_out(&sum1, out1, saturationCount);
            saturate_store_out(&sum2, out2, saturationCount);
            saturate_store_out(&sum3, out3, saturationCount);
            saturate_store_out(&sum4, out4, saturationCount);
            saturate_store_out(&sum5, out5, saturationCount);
            saturate_store_out(&sum6, out6, saturationCount);
            saturate_store_out(&sum7, out7, saturationCount);
            saturate_store_out(&sum8, out8, saturationCount);

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

            saturate_store_out(&sum1, out1++, saturationCount);
        }
    }
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

    mm_ptr in1, in2, in3, in4, in5, in6, flt, in_end;
    mm_vector v1, v2, v3, v4, v5, v6, f;
    mm_vector im1, im2, im3, im4, im5, im6;
    mm_vector acc1, acc2, acc3, acc4, acc5, acc6;

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

                    sum1 += vec_sum(acc1);

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum1;
                }
            }
            pool_end_index += inc;
            pool_end_index %= PS;


            j += inc;
            pool_num_entries += inc;
            if (static_cast<uint32_t>(pool_num_entries) == PS)
            {
                for (i = 0; i < FN; i++)
                {
                    func_partial_pooling(PS, PS, 0, pool + i * CNN_POOL_SIZE_MAX, &value);
                    gna_saturate_cast(value, *saturationCount);
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
            gna_saturate_cast(value, *saturationCount);

            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}

using m128i_x2 = decltype(std::make_pair(_mm_setzero_si128(), _mm_setzero_si128()));
using m128i_x4 = decltype(std::make_pair(m128i_x2{}, m128i_x2{}));
/* Multiply and add 2*sizeof(__m128i) elements, supports 8 and 16bit types
 * possibly zero elements that are past data bound (if apply_mask is set),
 * mask should have one or two elements (the latter only in case of "2b2b").
 *
 * Steps:
 * extend both input and filter from epi8 to epi16 for multiplication (when used with 8bit types)
 * multiply them
 * (elements zeroed previously in input now yield zeros, so would not bother later accumulation)
 * (use _mm_madd_epi16() instead of _mm_mullo_epi16(), so we get one level of horizontal sum
 * for free, it also expands data to 32bit, as would be needed in accumulation anyway).
 *
 * Returns as four registers to decrease dependency chain between (outer) loop steps
 *
 * Compilation note:
 * versions of GCC prior to 7 and ICC prior to 19 do not support constexpr if.
 * However marking appropiate conditions as constexpr if do not result in different code,
 * at least when comparing between ICC 18 and ICC 19 on Linux.
 */
template <typename filter_t, typename input_t>
static inline m128i_x4 madd_32_elems(filter_t *filter, input_t *input, __m128i *mask, bool apply_mask)
{
    const __m128i *F = (const __m128i *) filter;
    const __m128i *I = (const __m128i *) input;
    __m128i filter_0 = _mm_loadu_si128(F++);
    __m128i input_0 = _mm_loadu_si128(I++);
    __m128i filter_1, input_1;
    if (sizeof(filter_t) == 2) {
        filter_1 = _mm_loadu_si128(F++);
    }
    if (sizeof(input_t) == 2) {
        input_1 = _mm_loadu_si128(I++);
    }
    __m128i filter_2 = _mm_loadu_si128(F++);
    __m128i input_2 = _mm_loadu_si128(I++);
    __m128i filter_3, input_3;
    if (sizeof(filter_t) == 2) {
        filter_3 = _mm_loadu_si128(F++);
    }
    if (sizeof(input_t) == 2) {
        input_3 = _mm_loadu_si128(I++);
    }
    if (apply_mask) {
        if (sizeof(filter_t) <= sizeof(input_t)) {
            filter_0 = _mm_and_si128(filter_0, mask[0]);
            if (sizeof(filter_t) == 2) {
                filter_1 = _mm_and_si128(filter_1, mask[1]);
                filter_2 = _mm_and_si128(filter_2, mask[2]);
                filter_3 = _mm_and_si128(filter_3, mask[3]);
            } else {
                filter_2 = _mm_and_si128(filter_2, mask[1]);
            }
        }
        else {
            input_0 = _mm_and_si128(input_0, mask[0]);
            input_2 = _mm_and_si128(input_2, mask[1]);
        }
    }
    if (sizeof(filter_t) == 1) {
        filter_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(filter_0, 8));
        filter_0 = _mm_cvtepi8_epi16(filter_0);
    }
    if (sizeof(input_t) == 1) {
        input_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(input_0, 8));
        input_0 = _mm_cvtepi8_epi16(input_0);
    }
    if (sizeof(filter_t) == 1) {
        filter_3 = _mm_cvtepi8_epi16(_mm_bsrli_si128(filter_2, 8));
        filter_2 = _mm_cvtepi8_epi16(filter_2);
    }
    if (sizeof(input_t) == 1) {
        input_3 = _mm_cvtepi8_epi16(_mm_bsrli_si128(input_2, 8));
        input_2 = _mm_cvtepi8_epi16(input_2);
    }
    __m128i m0 = _mm_madd_epi16(input_0, filter_0);
    __m128i m1 = _mm_madd_epi16(input_1, filter_1);
    __m128i m2 = _mm_madd_epi16(input_2, filter_2);
    __m128i m3 = _mm_madd_epi16(input_3, filter_3);
    return { {m0, m1}, {m2, m3} };
}

template<typename T, size_t N>
static constexpr std::array<T, N> initByHalves(T firstHalfVal, T secondHalfVal) {
	std::array<T, N> a = {};
	size_t i = 0;
	for (; i < N/2; ++i) {
		a[i] = firstHalfVal;
	}
	for (; i < N; ++i) {
		a[i] = secondHalfVal;
	}
	return a;
}

template <typename filter_t, typename input_t>
static void cnn2d(ExecutionKernelConfig<ConvolutionConfig2D> const *const config)
{
    static_assert(sizeof(filter_t) <= 2 && sizeof(input_t) <= 2,
                  "filter_t and input_t must be int8_t or int16_t");
    const auto conf = config->RequestConfig;
    const input_t *const I = (input_t *)conf->Inputs;
    const filter_t *const F = (filter_t *)conf->Transform.FilterData;
    int32_t *O = (int32_t *)conf->Outputs;

    uint32_t inputDepth = conf->Transform.InputDepth;
    uint32_t inputHeight = conf->Transform.InputHeight;
    uint32_t inputWidth = conf->Transform.InputWidth;

    uint32_t numFilters = conf->Transform.NumberOfFilters;
    uint32_t filterHeight = conf->Transform.FilterHeight;
    uint32_t filterWidth = conf->Transform.FilterWidth;
    constexpr uint32_t sizeof_filter_t = sizeof(filter_t);
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth * sizeof_filter_t);
    uint32_t filterPadding = (Gna2RoundUp(memForFilter, 16) - memForFilter) / sizeof_filter_t;

    uint32_t padHeight = conf->Transform.ZeroPaddingHeight;
    uint32_t padWidth = conf->Transform.ZeroPaddingWidth;
    uint32_t strideHeight = conf->Transform.StrideHeight;
    uint32_t strideWidth = conf->Transform.StrideWidth;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    auto biasMode = conf->Transform.BiasMode;
    auto biasPrecission = conf->Transform.BiasDataMode;
    const void *biasData = conf->Transform.BiasData;

    // algo moves by the same number of elements per one step, irrelevant of input/filter width
    // (so it moves by twice as many bytes in 2B case vs 1B case)
    constexpr const uint32_t elems = sizeof(__m128i) / sizeof(int16_t);
    constexpr const uint32_t step = elems * 2 * 2; // how many elems are processed per loop step
    constexpr const bool is_2b2b = sizeof(filter_t) == 2 && sizeof(input_t) == 2;
    using mask_t = typename std::conditional<is_2b2b, int16_t, int8_t>::type;
    const auto maskArray = initByHalves<mask_t, step*2>(-1, 0);
    const auto mask = maskArray.data();

    for (uint32_t OD = 0; OD < numFilters; OD++) {
        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

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
                    outVal = getBias(biasData, biasPrecission,
                                     numFilters * outWidth * OH + numFilters * OW + OD);
                }

                /* Thanks to the fact that data is packed, we could iterate over W and Z dimensions via one loop.
                 * This observation improves performance a lot for case when W*Z is big, but one of dims is small. */
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
                const uint32_t stepsRounded = Gna2RoundUp(stepsPerWxZ, step);
                uint32_t stepIH = inputDepth * inputWidth - stepsRounded;
                uint32_t stepFH = inputDepth * filterWidth - stepsRounded;
                __m128i acc_0 = _mm_setzero_si128();
                __m128i acc_1 = _mm_setzero_si128();
                __m128i acc_2 = _mm_setzero_si128();
                __m128i acc_3 = _mm_setzero_si128();
                __m128i masked[4];
                masked[0] = _mm_loadu_si128(0+(const __m128i *)(mask + step - stepsPerWxZ % step));
                masked[1] = _mm_loadu_si128(1+(const __m128i *)(mask + step - stepsPerWxZ % step));
                if (is_2b2b) {
                    masked[2] = _mm_loadu_si128(2+(const __m128i *)(mask + step - stepsPerWxZ % step));
                    masked[3] = _mm_loadu_si128(3+(const __m128i *)(mask + step - stepsPerWxZ % step));
                }
                for (; steps--; idxF += stepFH, idxI += stepIH) {
                    for (uint32_t i = 0; i < stepsPerWxZ; i += step, idxF += step, idxI += step) {
                        auto m = madd_32_elems(F + idxF, I + idxI, masked, (i + step > stepsPerWxZ));
                        acc_0 = _mm_add_epi32(acc_0, m.first.first);
                        acc_1 = _mm_add_epi32(acc_1, m.first.second);
                        acc_2 = _mm_add_epi32(acc_2, m.second.first);
                        acc_3 = _mm_add_epi32(acc_3, m.second.second);
                    }
                }
                acc_0 = _mm_add_epi32(acc_0, acc_1);
                acc_2 = _mm_add_epi32(acc_2, acc_3);
                acc_0 = _mm_add_epi32(acc_0, acc_2);
                outVal += _mm_hsum_epi32(acc_0);
                gna_saturate_cast(outVal, *config->SaturationCount);
                O[numFilters * outWidth * OH + numFilters * OW + OD] = (int32_t)outVal;
            }
        }
    }
}

void Convolution2DKernelImpl1B1B(ExecutionKernelConfig<ConvolutionConfig2D> const *const config)
{
    return cnn2d<int8_t, int8_t>(config);
}

void Convolution2DKernelImpl1B2B(ExecutionKernelConfig<ConvolutionConfig2D> const *const config)
{
    return cnn2d<int8_t, int16_t>(config);
}

void Convolution2DKernelImpl2B1B(ExecutionKernelConfig<ConvolutionConfig2D> const *const config)
{
    return cnn2d<int16_t, int8_t>(config);
}

void Convolution2DKernelImpl2B2B(ExecutionKernelConfig<ConvolutionConfig2D> const *const config)
{
    return cnn2d<int16_t, int16_t>(config);
}

template <typename data_t>
static void poolMax2d(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    const auto conf = config->RequestConfig;
    data_t *I = (data_t *)conf->Inputs;
    data_t *O = (data_t *)conf->Outputs;

    uint32_t inputW = conf->Transform.InputWidth;
    uint32_t inputH = conf->Transform.InputHeight;
    uint32_t numFilters = conf->Transform.InputDepth;

    uint32_t poolStrideH = conf->Transform.StrideHeight;
    uint32_t poolStrideW = conf->Transform.StrideWidth;
    uint32_t windowHeight = conf->Transform.WindowHeight;
    uint32_t windowWidth = conf->Transform.WindowWidth;

    uint32_t wDimPartial = (inputW < windowWidth) ? 0 : inputW - windowWidth;
    uint32_t hDimPartial = (inputH < windowHeight) ? 0 : inputH - windowHeight;
    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(wDimPartial) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(hDimPartial) / (float)poolStrideH);

    constexpr const bool is1B = (sizeof(data_t) == 1);
    constexpr const bool is2B = (sizeof(data_t) == 2);
    constexpr const bool is4B = (sizeof(data_t) == 4);
    constexpr const uint32_t elems = sizeof(__m128i) / sizeof(data_t);
    constexpr const uint32_t step = 2*elems;
    constexpr data_t minLimit = (std::numeric_limits<data_t>::min)();
    constexpr data_t maxLimit = (std::numeric_limits<data_t>::max)();
    const auto maskArray = initByHalves<data_t, step*2>(maxLimit, minLimit);
    if (is1B) {
        memset(O, minLimit, poolOutH * poolOutW * numFilters);
    }
    else {
        std::fill(O, O + poolOutH * poolOutW * numFilters, minLimit);
    }

    const __m128i mask0 = _mm_loadu_si128((const __m128i *)(maskArray.data() + step - numFilters % step));
    const __m128i mask1 = _mm_loadu_si128((const __m128i *)(maskArray.data() + step + elems - numFilters % step));

    for (uint32_t POH = 0; POH < poolOutH; POH++) {
        uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;

        for (uint32_t POW = 0; POW < poolOutW; POW++) {
            uint32_t inIdxW = numFilters * POW * poolStrideW;

            uint32_t limH = windowHeight;
            if (POH * poolStrideH > inputH) {
                limH = 0;
            }
            else if (inputH - POH * poolStrideH < limH) {
                limH = inputH - POH * poolStrideH;
            }
            for (uint32_t OH = 0; OH < limH; OH++) {
                uint32_t winIdxH = numFilters * inputW * OH;

                uint32_t limW = windowWidth;
                if (POW * poolStrideW > inputW) {
                    limW = 0;
                }
                else if (inputW - POW * poolStrideW < limW) {
                    limW = inputW - POW * poolStrideW;
                }
                for (uint32_t OW = 0; OW < limW; OW++) {
                    uint32_t winIdxW = numFilters * OW;

                    uint32_t outBaseIdx = POH * poolOutW * numFilters + POW * numFilters;
                    uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                    uint32_t offset = 0;
                    // apply O[i] = max(O[i], I[i])
                    for (; offset < numFilters; offset += step) {
                        __m128i in0  = _mm_loadu_si128(0+(const __m128i *)(I + inBaseIdx + offset));
                        __m128i in1  = _mm_loadu_si128(1+(const __m128i *)(I + inBaseIdx + offset));
                        __m128i cur0 = _mm_loadu_si128(0+(const __m128i *)(O + outBaseIdx + offset));
                        __m128i cur1 = _mm_loadu_si128(1+(const __m128i *)(O + outBaseIdx + offset));
                        __m128i mx0, mx1;
                        if (is1B) {
                            if (offset + step > numFilters) {
                                in0 = _mm_min_epi8(in0, mask0);
                                in1 = _mm_min_epi8(in1, mask1);
                            }
                            mx0 = _mm_max_epi8(cur0, in0);
                            mx1 = _mm_max_epi8(cur1, in1);
                        }
                        if (is2B) {
                            if (offset + step > numFilters) {
                                in0 = _mm_min_epi16(in0, mask0);
                                in1 = _mm_min_epi16(in1, mask1);
                            }
                            mx0 = _mm_max_epi16(cur0, in0);
                            mx1 = _mm_max_epi16(cur1, in1);
                        }
                        if (is4B) {
                            if (offset + step > numFilters) {
                                in0 = _mm_min_epi32(in0, mask0);
                                in1 = _mm_min_epi32(in1, mask1);
                            }
                            mx0 = _mm_max_epi32(cur0, in0);
                            mx1 = _mm_max_epi32(cur1, in1);
                        }
                        _mm_storeu_si128(0+(__m128i *)(O + outBaseIdx + offset), mx0);
                        _mm_storeu_si128(1+(__m128i *)(O + outBaseIdx + offset), mx1);
                    }
                }
            }
        }
    }
}

template <typename data_t>
static void poolSum2d(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    const auto conf = config->RequestConfig;
    data_t *I = (data_t *)conf->Inputs;
    data_t *O = (data_t *)conf->Outputs;

    uint32_t inputW = conf->Transform.InputWidth;
    uint32_t inputH = conf->Transform.InputHeight;
    uint32_t numFilters = conf->Transform.InputDepth;

    uint32_t poolStrideH = conf->Transform.StrideHeight;
    uint32_t poolStrideW = conf->Transform.StrideWidth;
    uint32_t windowHeight = conf->Transform.WindowHeight;
    uint32_t windowWidth = conf->Transform.WindowWidth;

    uint32_t wDimPartial = (inputW < windowWidth) ? 0 : inputW - windowWidth;
    uint32_t hDimPartial = (inputH < windowHeight) ? 0 : inputH - windowHeight;
    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(wDimPartial) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(hDimPartial) / (float)poolStrideH);

    constexpr const bool is1B = (sizeof(data_t) == 1);
    constexpr const bool is2B = (sizeof(data_t) == 2);
    constexpr const bool is4B = (sizeof(data_t) == 4);
    constexpr const uint32_t elems = sizeof(__m128i) / sizeof(data_t);
    constexpr const uint32_t step = elems * 1 * 2;  // how many elems are processed per loop step
    __m128i satCntHighBit00 = _mm_setzero_si128();
    __m128i satCntHighBit01 = _mm_setzero_si128();
    __m128i satCntAllBits00 = _mm_setzero_si128();
    __m128i satCntAllBits01 = _mm_setzero_si128();
    __m128i satCntHighBit10 = _mm_setzero_si128();
    __m128i satCntHighBit11 = _mm_setzero_si128();
    __m128i satCntAllBits10 = _mm_setzero_si128();
    __m128i satCntAllBits11 = _mm_setzero_si128();

    /* Following variable is used in 1B variant, to avoid bumping of SaturationCount when it is not necessary.
     * Calculations are still performed saturation-aware, it is only the counter that is not touched.
     * This approach yields about 14% improvement over naive version (if-less, with all instructions of "if" path).
     *
     * Similar trick makes no sense in 2B variant, where satCnt is calculated with single vectorized "or".
     */
    uint32_t needsSatCnt = 1;

    for (uint32_t POH = 0; POH < poolOutH; POH++) {
        uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;

        for (uint32_t POW = 0; POW < poolOutW; POW++) {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            uint32_t outBaseIdx = POH * poolOutW * numFilters + POW * numFilters;

            uint32_t limW = windowWidth;
            if (POW * poolStrideW > inputW) {
                limW = 0;
            }
            else if (inputW - POW * poolStrideW < limW) {
                limW = inputW - POW * poolStrideW;
            }
            uint32_t limH = windowHeight;
            if (POH * poolStrideH > inputH) {
                limH = 0;
            }
            else if (inputH - POH * poolStrideH < limH) {
                limH = inputH - POH * poolStrideH;
            }

            // usage of following variable SLOWS up 2B case by ~8.5%
            // TODO: check with 2x unroll and variable off
            //       (because 2x unroll with this variable present is current best case)
            const bool couldEverOV = (limW * limH * numFilters > 0x10000);
            uint32_t couldOV = 0;
            for (uint32_t offset = 0; offset < numFilters; offset += step) {
                __m128i cur00 = _mm_setzero_si128();
                __m128i cur01 = _mm_setzero_si128();
                __m128i cur10 = _mm_setzero_si128();
                __m128i cur11 = _mm_setzero_si128();
                for (uint32_t OW = 0; OW < limW; OW++) {
                    uint32_t winIdxW = numFilters * OW;
                    for (uint32_t OH = 0; OH < limH; OH++) {
                        uint32_t winIdxH = numFilters * inputW * OH;

                        uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                        __m128i in0 = _mm_loadu_si128((const __m128i *)(I + inBaseIdx + offset));
                        __m128i in1 = _mm_loadu_si128((const __m128i *)(I + inBaseIdx + offset + elems));
                        if (is1B) {
                            __m128i in01 = _mm_cvtepi8_epi16(_mm_bsrli_si128(in0, 8));
                            __m128i in00 = _mm_cvtepi8_epi16(in0);
                            __m128i in11 = _mm_cvtepi8_epi16(_mm_bsrli_si128(in1, 8));
                            __m128i in10 = _mm_cvtepi8_epi16(in1);
                            if ((couldOV += needsSatCnt) > 0x100) {
                                __m128i noSat00 = _mm_add_epi16(cur00, in00);
                                __m128i noSat01 = _mm_add_epi16(cur01, in01);
                                __m128i noSat10 = _mm_add_epi16(cur10, in10);
                                __m128i noSat11 = _mm_add_epi16(cur11, in11);
                                cur00 = _mm_adds_epi16(cur00, in00);
                                cur01 = _mm_adds_epi16(cur01, in01);
                                cur10 = _mm_adds_epi16(cur10, in10);
                                cur11 = _mm_adds_epi16(cur11, in11);
                                __m128i x00 = _mm_xor_si128(noSat00, cur00);
                                __m128i x01 = _mm_xor_si128(noSat01, cur01);
                                __m128i x10 = _mm_xor_si128(noSat10, cur10);
                                __m128i x11 = _mm_xor_si128(noSat11, cur11);
                                satCntAllBits00 = _mm_or_si128(satCntAllBits00, x00);
                                satCntAllBits01 = _mm_or_si128(satCntAllBits01, x01);
                                satCntAllBits10 = _mm_or_si128(satCntAllBits10, x10);
                                satCntAllBits11 = _mm_or_si128(satCntAllBits11, x11);
                            }
                            else {
                                cur00 = _mm_adds_epi16(cur00, in00);
                                cur01 = _mm_adds_epi16(cur01, in01);
                                cur10 = _mm_adds_epi16(cur10, in10);
                                cur11 = _mm_adds_epi16(cur11, in11);
                            }
                        }
                        if (is2B) {
                            __m128i in01 = _mm_cvtepi16_epi32(_mm_bsrli_si128(in0, 8));
                            __m128i in00 = _mm_cvtepi16_epi32(in0);
                            __m128i in11 = _mm_cvtepi16_epi32(_mm_bsrli_si128(in1, 8));
                            __m128i in10 = _mm_cvtepi16_epi32(in1);
                            if (couldEverOV && ++couldOV > 0x10000) {
                                cur00 = _mm_adds_epi32(cur00, in00, &satCntHighBit00);
                                cur01 = _mm_adds_epi32(cur01, in01, &satCntHighBit01);
                                cur10 = _mm_adds_epi32(cur10, in10, &satCntHighBit10);
                                cur11 = _mm_adds_epi32(cur11, in11, &satCntHighBit11);
                            }
                            else {
                                cur00 = _mm_add_epi32(cur00, in00);
                                cur01 = _mm_add_epi32(cur01, in01);
                                cur10 = _mm_add_epi32(cur10, in10);
                                cur11 = _mm_add_epi32(cur11, in11);
                            }
                        }
                        if (is4B) {
                            cur00 = _mm_adds_epi32(cur00, in0, &satCntHighBit00);
                            cur10 = _mm_adds_epi32(cur10, in1, &satCntHighBit10);
                        }
                    }
                }
                if (is4B) {
                    if (offset + elems > numFilters) {
                        data_t out[elems];
                        _mm_storeu_si128((__m128i *)out, cur00);
                        memcpy(O + outBaseIdx + offset, out, sizeof(data_t) * (numFilters % elems));
                    }
                    else if (offset + step > numFilters) {
                        data_t out[elems];
                        _mm_storeu_si128((__m128i *)out, cur10);
                        _mm_storeu_si128((__m128i *)(O + outBaseIdx + offset), cur00);
                        memcpy(O + outBaseIdx + offset + elems, out, sizeof(data_t) * (numFilters % elems));
                    } else {
                        _mm_storeu_si128((__m128i *)(O + outBaseIdx + offset), cur00);
                        _mm_storeu_si128((__m128i *)(O + outBaseIdx + offset + elems), cur10);
                    }
                }
                else {
                    __m128i ret0, ret00, ret01;
                    __m128i ret1, ret10, ret11;
                    if (is1B) {
                        ret0 = _mm_packs_epi16(cur00, cur01);
                        ret1 = _mm_packs_epi16(cur10, cur11);
                        if (needsSatCnt) {
                            ret01 = _mm_cvtepi8_epi16(_mm_bsrli_si128(ret0, 8));
                            ret00 = _mm_cvtepi8_epi16(ret0);
                            ret11 = _mm_cvtepi8_epi16(_mm_bsrli_si128(ret1, 8));
                            ret10 = _mm_cvtepi8_epi16(ret1);
                        }
                    }
                    if (is2B) {
                        ret0 = _mm_packs_epi32(cur00, cur01);
                        ret1 = _mm_packs_epi32(cur10, cur11);
                        ret01 = _mm_cvtepi16_epi32(_mm_bsrli_si128(ret0, 8));
                        ret00 = _mm_cvtepi16_epi32(ret0);
                        ret11 = _mm_cvtepi16_epi32(_mm_bsrli_si128(ret1, 8));
                        ret10 = _mm_cvtepi16_epi32(ret1);
                    }
                    if (needsSatCnt) {
                        __m128i x01 = _mm_xor_si128(ret01, cur01);
                        __m128i x00 = _mm_xor_si128(ret00, cur00);
                        __m128i x11 = _mm_xor_si128(ret11, cur11);
                        __m128i x10 = _mm_xor_si128(ret10, cur10);
                        satCntAllBits00 = _mm_or_si128(satCntAllBits00, x00);
                        satCntAllBits01 = _mm_or_si128(satCntAllBits01, x01);
                        satCntAllBits10 = _mm_or_si128(satCntAllBits10, x10);
                        satCntAllBits11 = _mm_or_si128(satCntAllBits11, x11);
                        if (is1B) {
                            if (        _mm_test_any(satCntAllBits00) || _mm_test_any(satCntAllBits01)
                                     || _mm_test_any(satCntAllBits10) || _mm_test_any(satCntAllBits11)
                            ) {
                                needsSatCnt = 0;
                            }
                        }
                    }
                    if (offset + elems > numFilters) {
                        data_t out[elems];
                        _mm_storeu_si128((__m128i *)out, ret0);
                        memcpy(O + outBaseIdx + offset, out, sizeof(data_t) * (numFilters % elems));
                    }
                    else if (offset + step > numFilters) {
                        data_t out[elems];
                        _mm_storeu_si128((__m128i *)out, ret1);
                        _mm_storeu_si128((__m128i *)(O + outBaseIdx + offset), ret0);
                        memcpy(O + outBaseIdx + offset + elems, out, sizeof(data_t) * (numFilters % elems));
                    }
                    else {
                        _mm_storeu_si128((__m128i *)(O + outBaseIdx + offset), ret0);
                        _mm_storeu_si128((__m128i *)(O + outBaseIdx + offset + elems), ret1);
                    }
                }
            }
        }
    }
    if (!is1B) {
        *config->SaturationCount += _mm_test_anyMSB_epi32(satCntHighBit00);
        *config->SaturationCount += _mm_test_anyMSB_epi32(satCntHighBit01);
    }
    if (is2B) {
        *config->SaturationCount += _mm_test_anyMSB_epi32(satCntHighBit10);
        *config->SaturationCount += _mm_test_anyMSB_epi32(satCntHighBit11);
    }
    if (!is4B) {
        *config->SaturationCount += _mm_test_any(satCntAllBits00);
        *config->SaturationCount += _mm_test_any(satCntAllBits01);
        *config->SaturationCount += _mm_test_any(satCntAllBits10);
        *config->SaturationCount += _mm_test_any(satCntAllBits11);
    }
}

/* This is a specialized version of pooling Sum.
 * Before 2x unrolling (step=elems*2) it is 1.25x times faster than non-specialized version because it uses different algorithm.
 * 2x unroll yields another 1.2x improvement (TODO: check non-specialized version with 2x unroll).
 *
 * Main difference is loop order,
 * what means that we compute each output value via many LOAD/ADDS/STORE cycles.
 * Thanks to the fact above we could traverse input in more favorable way memory-wise. */
template <>
void poolSum2d<int32_t>(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    const auto conf = config->RequestConfig;
    using data_t = int32_t;
    data_t *I = (data_t *)conf->Inputs;
    data_t *O = (data_t *)conf->Outputs;

    uint32_t inputW = conf->Transform.InputWidth;
    uint32_t inputH = conf->Transform.InputHeight;
    uint32_t numFilters = conf->Transform.InputDepth;

    uint32_t poolStrideH = conf->Transform.StrideHeight;
    uint32_t poolStrideW = conf->Transform.StrideWidth;
    uint32_t windowHeight = conf->Transform.WindowHeight;
    uint32_t windowWidth = conf->Transform.WindowWidth;

    uint32_t wDimPartial = (inputW < windowWidth) ? 0 : inputW - windowWidth;
    uint32_t hDimPartial = (inputH < windowHeight) ? 0 : inputH - windowHeight;
    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(wDimPartial) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(hDimPartial) / (float)poolStrideH);

    constexpr const uint32_t elems = sizeof(__m128i) / sizeof(data_t);
    constexpr const uint32_t step = elems * 1 * 2;  // how many elems are processed per loop step
    memset(O, 0, poolOutH * poolOutW * numFilters * sizeof(data_t));  // we are calculating out via many parial sums
    const auto maskArray = initByHalves<data_t, step*2>(-1, 0);
    const __m128i mask0 = _mm_loadu_si128((const __m128i *)(maskArray.data() + step - numFilters % step));
    const __m128i mask1 = _mm_loadu_si128((const __m128i *)(maskArray.data() + step + elems - numFilters % step));
    __m128i satCnt0 = _mm_setzero_si128();
    __m128i satCnt1 = _mm_setzero_si128();

    for (uint32_t POH = 0; POH < poolOutH; POH++) {
        uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;

        for (uint32_t POW = 0; POW < poolOutW; POW++) {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            uint32_t outBaseIdx = POH * poolOutW * numFilters + POW * numFilters;

            uint32_t limW = windowWidth;
            if (POW * poolStrideW > inputW) {
                limW = 0;
            }
            else if (inputW - POW * poolStrideW < limW) {
                limW = inputW - POW * poolStrideW;
            }
            uint32_t limH = windowHeight;
            if (POH * poolStrideH > inputH) {
                limH = 0;
            }
            else if (inputH - POH * poolStrideH < limH) {
                limH = inputH - POH * poolStrideH;
            }

            for (uint32_t OW = 0; OW < limW; OW++) {
                uint32_t winIdxW = numFilters * OW;
                for (uint32_t OH = 0; OH < limH; OH++) {
                    uint32_t winIdxH = numFilters * inputW * OH;

                    uint32_t inBaseIdx = inIdxW + inIdxH + winIdxW + winIdxH;
                    for (uint32_t offset = 0; offset < numFilters; offset += step) {
                        __m128i in0  = _mm_loadu_si128(0+(const __m128i *)(I + inBaseIdx + offset));
                        __m128i in1  = _mm_loadu_si128(1+(const __m128i *)(I + inBaseIdx + offset));
                        __m128i cur0 = _mm_loadu_si128(0+(const __m128i *)(O + outBaseIdx + offset));
                        __m128i cur1 = _mm_loadu_si128(1+(const __m128i *)(O + outBaseIdx + offset));
                        if (offset + step > numFilters) {
                            in0 = _mm_and_si128(in0, mask0);
                            in1 = _mm_and_si128(in1, mask1);
                        }
                        __m128i ret0 = _mm_adds_epi32(cur0, in0, &satCnt0);
                        __m128i ret1 = _mm_adds_epi32(cur1, in1, &satCnt1);
                        _mm_storeu_si128(0+(__m128i *)(O + outBaseIdx + offset), ret0);
                        _mm_storeu_si128(1+(__m128i *)(O + outBaseIdx + offset), ret1);
                    }
                }
            }
        }
    }
    *config->SaturationCount += _mm_test_anyMSB_epi32(satCnt0);
    *config->SaturationCount += _mm_test_anyMSB_epi32(satCnt1);
}

template <typename data_t>
static inline void pool2d(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    auto mode = config->RequestConfig->Transform.Mode;
    if (mode == KernelPoolingModeMax) {
        poolMax2d<data_t>(config);
    }
    else if (mode == KernelPoolingModeSum) {
        poolSum2d<data_t>(config);
    }
}

void Pooling2DKernelImpl1B(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    pool2d<int8_t>(config);
}

void Pooling2DKernelImpl2B(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    pool2d<int16_t>(config);
}

void Pooling2DKernelImpl4B(ExecutionKernelConfig<PoolingConfig2D> const *const config)
{
    pool2d<int32_t>(config);
}
