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

// TODO: make naming convention consistent with other kernel implementations

#include "convnet.h"
#include "igemv.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cmath>
#include <cstdint>

__forceinline void saturate64_store_out(int64_t * const out, uint32_t * const saturationCount)
{
    if (*out > INT32_MAX)
    {
        *out = INT32_MAX;
        (*saturationCount)++;
    }
    else if (*out < INT32_MIN)
    {
        *out = INT32_MIN;
        (*saturationCount)++;
    }
}

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
    const int8_t * const B = (int8_t *)filterConfig->biases;
    int32_t * const O = filterConfig->convolutedOutputs;

    uint32_t i;
    uint32_t j;
    uint32_t k;

    gna_sum_t sum = 0;

    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;


    const int16_t* ptr_coef;
    const int16_t* ptr_in;
    for (j = 0; j < num_filter_outputs; j++)
    {
        ptr_in = I + j * num_inputs_band_stride;
        for (i = 0; i < FN; i++)
        {
            ptr_coef = F + i * FC;

            if (filterConfig->bytesPerBias == 1)
            {
                sum = B[i];
            }
            else if (filterConfig->bytesPerBias == 2)
            {
                sum = ((int16_t*)B)[i];
            }
            else if (filterConfig->bytesPerBias == 4)
            {
                sum = ((int32_t*)B)[i];
            }

            for (k = 0; k < FC; k++)
            {
                sum += ptr_in[k] * ptr_coef[k];
            }
            O[j * FN + i] = sum;
        }
    }
}

void ConvolutionKernelImpl1B(ConvolutionConfig const * const filterConfig)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int8_t* const I = (int8_t*)filterConfig->inputs;
    const int8_t* const F = (int8_t*)filterConfig->filters;
    const int8_t * const B = (int8_t *)filterConfig->biases;
    int32_t * const O = filterConfig->convolutedOutputs;

    uint32_t i;
    uint32_t j;
    uint32_t k;

    gna_sum_t sum = 0;

    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;


    const int8_t* ptr_coef;
    const int8_t* ptr_in;
    for (j = 0; j < num_filter_outputs; j++)
    {
        ptr_in = I + j * num_inputs_band_stride;
        for (i = 0; i < FN; i++)
        {
            ptr_coef = F + i * FC* filterConfig->bytesPerFilter;

            if (filterConfig->bytesPerBias == 1)
            {
                sum = B[i];
            }
            else if (filterConfig->bytesPerBias == 2)
            {
                sum = ((int16_t*)B)[i];
            }
            else if (filterConfig->bytesPerBias == 4)
            {
                sum = ((int32_t*)B)[i];
            }

            for (k = 0; k < FC; k++)
            {
                if (filterConfig->bytesPerFilter == 1)
                {
                    sum += ptr_in[k] * ptr_coef[k];
                }
                else if (filterConfig->bytesPerFilter == 2)
                {
                    sum += ptr_in[k] * ((int16_t*)ptr_coef)[k];
                }
            }
            O[j * FN + i] = sum;
        }
    }
}

void ConvolutionKernelImpl2B(ConvolutionConfig const * const filterConfig)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int16_t* const I = filterConfig->inputs;
    const int8_t* const F = (int8_t*)filterConfig->filters;
    const int8_t * const B = (int8_t *)filterConfig->biases;
    int32_t * const O = filterConfig->convolutedOutputs;

    uint32_t i;
    uint32_t j;
    uint32_t k;

    gna_sum_t sum = 0;

    uint32_t num_inputs_band_stride = filterConfig->inputBandStride;
    uint32_t num_filter_outputs = filterConfig->filterOutputCount;


    const int8_t* ptr_coef;
    const int16_t* ptr_in;
    for (j = 0; j < num_filter_outputs; j++)
    {
        ptr_in = I + j * num_inputs_band_stride;
        for (i = 0; i < FN; i++)
        {
            ptr_coef = F + i * FC* filterConfig->bytesPerFilter;

            if (filterConfig->bytesPerBias == 1)
            {
                sum = B[i];
            }
            else if (filterConfig->bytesPerBias == 2)
            {
                sum = ((int16_t*)B)[i];
            }
            else if (filterConfig->bytesPerBias == 4)
            {
                sum = ((int32_t*)B)[i];
            }

            for (k = 0; k < FC; k++)
            {
                if (filterConfig->bytesPerFilter == 1)
                {
                    sum += ptr_in[k] * ptr_coef[k];
                }
                else if (filterConfig->bytesPerFilter == 2)
                {
                    sum += ptr_in[k] * ((int16_t*)ptr_coef)[k];
                }
            }
            O[j * FN + i] = sum;
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
    const int8_t * const B = (int8_t *)filterConfig->biases;
    int16_t * const O = filterConfig->pooledOutputs;
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    const nn_pool_type PT = poolConfig->type;
    const uint32_t PS = poolConfig->size;
    const uint32_t PSTEP = poolConfig->step;
    int64_t * const pool = poolConfig->buffer;

    if (PS == 0)
    {
        return;
    }

    pwl->KERNEL(InitializeActivationFunctions)();

    void(*func_partial_pooling)(const uint32_t PS, const uint32_t pool_num_entries, const uint32_t pool_start_index, const int64_t* P, int64_t *V);

    if (PT == INTEL_SUM_POOLING)
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
    uint32_t k;
    const int16_t *ptr_in;
    const int16_t *ptr_coef;
    int64_t value;
    uint32_t inc;
    uint32_t l;
    gna_sum_t sum = 0;

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


            for (l = 0; l < inc; l++)
            {
                ptr_in = I + (j + l)*num_inputs_band_stride;

                for (i = 0; i < FN; i++)
                {
                    ptr_coef = F + i * FC;

                    if (filterConfig->bytesPerBias == 1)
                    {
                        sum = B[i];
                    }
                    else if (filterConfig->bytesPerBias == 2)
                    {
                        sum = ((int16_t*)B)[i];
                    }
                    else if (filterConfig->bytesPerBias == 4)
                    {
                        sum = ((int32_t*)B)[i];
                    }

                    for (k = 0; k < FC; k++)
                    {
                        sum += ptr_in[k] * ptr_coef[k];
                    }

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum;
                }

                pool_end_index = (pool_end_index + 1) % PS;
            }

            j += inc;
            pool_num_entries += inc;
            if (static_cast<uint32_t>(pool_num_entries) == PS)
            {
                for (i = 0; i < FN; i++)
                {
                    func_partial_pooling(PS, PS, 0, pool + i * CNN_POOL_SIZE_MAX, &value);
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
            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, &O[output_index * FN + i], saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}

void ConvolutionPoolingKernelImpl1B(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int8_t* const I = (int8_t *)filterConfig->inputs;
    const int8_t* const F = (int8_t *)filterConfig->filters;
    const int8_t * const B = (int8_t *)filterConfig->biases;
    int8_t * const O = (int8_t*)filterConfig->pooledOutputs;
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    const nn_pool_type PT = poolConfig->type;
    const uint32_t PS = poolConfig->size;
    const uint32_t PSTEP = poolConfig->step;
    int64_t * const pool = poolConfig->buffer;

    if (PS == 0)
    {
        return;
    }

    pwl->KERNEL(InitializeActivationFunctions)();

    void(*func_partial_pooling)(const uint32_t PS, const uint32_t pool_num_entries, const uint32_t pool_start_index, const int64_t* P, int64_t *V);

    if (PT == INTEL_SUM_POOLING)
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
    uint32_t k;
    const int8_t *ptr_in;
    const int8_t *ptr_coef;
    int64_t value;
    uint32_t inc;
    uint32_t l;
    gna_sum_t sum = 0;

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


            for (l = 0; l < inc; l++)
            {
                ptr_in = I + (j + l)*num_inputs_band_stride;

                for (i = 0; i < FN; i++)
                {
                    ptr_coef = F + i * FC * filterConfig->bytesPerFilter;

                    if (filterConfig->bytesPerBias == 1)
                    {
                        sum = B[i];
                    }
                    else if (filterConfig->bytesPerBias == 2)
                    {
                        sum = ((int16_t*)B)[i];
                    }
                    else if (filterConfig->bytesPerBias == 4)
                    {
                        sum = ((int32_t*)B)[i];
                    }

                    for (k = 0; k < FC; k++)
                    {
                        if (filterConfig->bytesPerFilter == 1)
                        {
                            sum += ptr_in[k] * ptr_coef[k];
                        }
                        else if (filterConfig->bytesPerFilter == 2)
                        {
                            sum += ptr_in[k] * ((int16_t*)ptr_coef)[k];
                        }
                    }

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum;
                }

                pool_end_index = (pool_end_index + 1) % PS;
            }

            j += inc;
            pool_num_entries += inc;
            if (static_cast<uint32_t>(pool_num_entries) == PS)
            {
                for (i = 0; i < FN; i++)
                {
                    func_partial_pooling(PS, PS, 0, pool + i * CNN_POOL_SIZE_MAX, &value);
                    pwl->ActivateSingle(&pwl->pwl, (int32_t)value, (int16_t*)&O[(output_index * FN + i) * pwl->pwl.bytesPerOutput], saturationCount);
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
            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, (int16_t*)&O[(output_index * FN + i) * pwl->pwl.bytesPerOutput], saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}

void ConvolutionPoolingKernelImpl2B(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl)
{
    const uint32_t FN = filterConfig->filterCount;
    const uint32_t FC = filterConfig->filterCoefficientCount;
    const int16_t* const I = filterConfig->inputs;
    const int8_t* const F = (int8_t *)filterConfig->filters;
    const int8_t * const B = (int8_t *)filterConfig->biases;
    int8_t * const O = (int8_t *)filterConfig->pooledOutputs;
    uint32_t * const saturationCount = filterConfig->execution->SaturationCount;

    const nn_pool_type PT = poolConfig->type;
    const uint32_t PS = poolConfig->size;
    const uint32_t PSTEP = poolConfig->step;
    int64_t * const pool = poolConfig->buffer;

    if (PS == 0)
    {
        return;
    }

    pwl->KERNEL(InitializeActivationFunctions)();

    void(*func_partial_pooling)(const uint32_t PS, const uint32_t pool_num_entries, const uint32_t pool_start_index, const int64_t* P, int64_t *V);

    if (PT == INTEL_SUM_POOLING)
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
    uint32_t k;
    const int16_t *ptr_in;
    const int8_t *ptr_coef;
    int64_t value;
    uint32_t inc;
    uint32_t l;
    gna_sum_t sum = 0;

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


            for (l = 0; l < inc; l++)
            {
                ptr_in = I + (j + l)*num_inputs_band_stride;

                for (i = 0; i < FN; i++)
                {
                    ptr_coef = F + i * FC * filterConfig->bytesPerFilter;

                    if (filterConfig->bytesPerBias == 1)
                    {
                        sum = B[i];
                    }
                    else if (filterConfig->bytesPerBias == 2)
                    {
                        sum = ((int16_t*)B)[i];
                    }
                    else if (filterConfig->bytesPerBias == 4)
                    {
                        sum = ((int32_t*)B)[i];
                    }

                    for (k = 0; k < FC; k++)
                    {
                        if (filterConfig->bytesPerFilter == 1)
                        {
                            sum += ptr_in[k] * ptr_coef[k];
                        }
                        else if (filterConfig->bytesPerFilter == 2)
                        {
                            sum += ptr_in[k] * ((int16_t*)ptr_coef)[k];
                        }
                    }

                    pool[i * CNN_POOL_SIZE_MAX + pool_end_index] = sum;
                }

                pool_end_index = (pool_end_index + 1) % PS;
            }

            j += inc;
            pool_num_entries += inc;
            if (static_cast<uint32_t>(pool_num_entries) == PS)
            {
                for (i = 0; i < FN; i++)
                {
                    func_partial_pooling(PS, PS, 0, pool + i * CNN_POOL_SIZE_MAX, &value);
                    pwl->ActivateSingle(&pwl->pwl, (int32_t)value, (int16_t*)&(O[(output_index * FN + i) * pwl->pwl.bytesPerOutput]), saturationCount);
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
            pwl->ActivateSingle(&pwl->pwl, (int32_t)value, (int16_t*)&(O[(output_index * FN + i) * pwl->pwl.bytesPerOutput]), saturationCount);
        }

        pool_start_index = (pool_start_index + PSTEP) % PS;
        pool_num_entries -= PSTEP;
        output_index++;
    }
}

void Pooling2DKernelImpl1B(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    int8_t* I = (int8_t*)config->RequestConfig->Inputs;
    int16_t* O = (int16_t*)config->RequestConfig->Outputs;

    uint32_t inputW = config->RequestConfig->Transform.Pooling.inputDimensions.width;
    uint32_t inputH = config->RequestConfig->Transform.Pooling.inputDimensions.height;
    uint32_t numFilters = config->RequestConfig->Transform.Pooling.inputDimensions.depth;

    uint32_t poolStrideH = config->RequestConfig->Transform.Pooling.pooling.stride.height;
    uint32_t poolStrideW = config->RequestConfig->Transform.Pooling.pooling.stride.width;
    uint32_t windowHeight = config->RequestConfig->Transform.Pooling.pooling.window.height;
    uint32_t windowWidth = config->RequestConfig->Transform.Pooling.pooling.window.width;

    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(inputW - windowWidth) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(inputH - windowHeight) / (float)poolStrideH);

    int64_t tmpValue = 0;
    int64_t value = 0;
    for (uint32_t OD = 0; OD < numFilters; OD++)
    {
        for (uint32_t POW = 0; POW < poolOutW; POW++)
        {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            for (uint32_t POH = 0; POH < poolOutH; POH++)
            {
                uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;
                tmpValue = 0;

                for (uint32_t OW = 0; OW < windowWidth; OW++)
                {
                    uint32_t winIdxW = numFilters * OW;

                    for (uint32_t OH = 0; OH < windowHeight; OH++) {

                        uint32_t winIdxH = numFilters * inputW * OH;

                        if (config->RequestConfig->Transform.Pooling.pooling.type == INTEL_MAX_POOLING)
                        {
                            if ((POW * poolStrideW <= (inputW - 1)) && (POH * poolStrideH <= (inputH - 1)))
                            {
                                tmpValue = I[OD + inIdxW + inIdxH];
                            }

                            if ((POW * poolStrideW + OW <= (inputW - 1)) && (POH * poolStrideH + OH <= (inputH - 1)))
                            {
                                tmpValue = I[OD + inIdxW + inIdxH + winIdxW + winIdxH];
                            }

                            if ((OW == 0) && (OH == 0))
                            {
                                value = tmpValue;
                            }
                            else if (value < tmpValue)
                            {
                                value = tmpValue;
                            }
                        }
                        else if (config->RequestConfig->Transform.Pooling.pooling.type == INTEL_SUM_POOLING)
                        {

                            if ((POW * poolStrideW + OW <= (inputW - 1)) && (POH * poolStrideH + OH <= (inputH - 1)))
                            {
                                tmpValue += I[OD + inIdxW + inIdxH + winIdxW + winIdxH];
                            }

                            if (value > 32767)
                            {
                                value = 32767;
                            }
                            else if (value < -32768)
                            {
                                value = -32768;
                            }
                            else
                            {
                                value = tmpValue;
                            }
                        }

                    }
                }

                O[POH * poolOutW * numFilters + POW * numFilters + OD] = (int16_t)value;
            }
        }
    }

}
void Pooling2DKernelImpl2B(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    int16_t* I = (int16_t*)config->RequestConfig->Inputs;
    int16_t* O = (int16_t*)config->RequestConfig->Outputs;

    uint32_t inputW = config->RequestConfig->Transform.Pooling.inputDimensions.width;
    uint32_t inputH = config->RequestConfig->Transform.Pooling.inputDimensions.height;
    uint32_t numFilters = config->RequestConfig->Transform.Pooling.inputDimensions.depth;

    uint32_t poolStrideH = config->RequestConfig->Transform.Pooling.pooling.stride.height;
    uint32_t poolStrideW = config->RequestConfig->Transform.Pooling.pooling.stride.width;
    uint32_t windowHeight = config->RequestConfig->Transform.Pooling.pooling.window.height;
    uint32_t windowWidth = config->RequestConfig->Transform.Pooling.pooling.window.width;


    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(inputW - windowWidth) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(inputH - windowHeight) / (float)poolStrideH);

    int64_t tmpValue = 0;
    int64_t value = 0;
    for (uint32_t OD = 0; OD < numFilters; OD++)
    {
        for (uint32_t POW = 0; POW < poolOutW; POW++)
        {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            for (uint32_t POH = 0; POH < poolOutH; POH++)
            {
                uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;
                tmpValue = 0;

                for (uint32_t OW = 0; OW < windowWidth; OW++)
                {
                    uint32_t winIdxW = numFilters * OW;

                    for (uint32_t OH = 0; OH < windowHeight; OH++) {

                        uint32_t winIdxH = numFilters * inputW * OH;

                        if (config->RequestConfig->Transform.Pooling.pooling.type == INTEL_MAX_POOLING)
                        {
                            if ((POW * poolStrideW <= (inputW - 1)) && (POH * poolStrideH <= (inputH - 1)))
                            {
                                tmpValue = I[OD + inIdxW + inIdxH];
                            }

                            if ((POW * poolStrideW + OW <= (inputW - 1)) && (POH * poolStrideH + OH <= (inputH - 1)))
                            {
                                tmpValue = I[OD + inIdxW + inIdxH + winIdxW + winIdxH];
                            }

                            if ((OW == 0) && (OH == 0))
                            {
                                value = tmpValue;
                            }
                            else if (value < tmpValue)
                            {
                                value = tmpValue;
                            }
                        }
                        else if (config->RequestConfig->Transform.Pooling.pooling.type == INTEL_SUM_POOLING)
                        {

                            if ((POW * poolStrideW + OW <= (inputW - 1)) && (POH * poolStrideH + OH <= (inputH - 1)))
                            {
                                tmpValue += I[OD + inIdxW + inIdxH + winIdxW + winIdxH];
                            }

                            if (value > 32767)
                            {
                                value = 32767;
                            }
                            else if (value < -32768)
                            {
                                value = -32768;
                            }
                            else
                            {
                                value = tmpValue;
                            }
                        }

                    }
                }

                O[POH * poolOutW * numFilters + POW * numFilters + OD] = (int16_t)value;
            }
        }
    }

}
void Pooling2DKernelImpl4B(ExecutionKernelConfig<PoolingConfig2D> const * const config)
{
    int32_t* I = (int32_t*)config->RequestConfig->Inputs;
    int32_t* O = (int32_t*)config->RequestConfig->Outputs;

    uint32_t inputW = config->RequestConfig->Transform.Pooling.inputDimensions.width;
    uint32_t inputH = config->RequestConfig->Transform.Pooling.inputDimensions.height;
    uint32_t numFilters = config->RequestConfig->Transform.Pooling.inputDimensions.depth;

    uint32_t poolStrideH = config->RequestConfig->Transform.Pooling.pooling.stride.height;
    uint32_t poolStrideW = config->RequestConfig->Transform.Pooling.pooling.stride.width;
    uint32_t windowHeight = config->RequestConfig->Transform.Pooling.pooling.window.height;
    uint32_t windowWidth = config->RequestConfig->Transform.Pooling.pooling.window.width;


    uint32_t poolOutW = 1 + (uint32_t)std::ceil((float)(inputW - windowWidth) / (float)poolStrideW);
    uint32_t poolOutH = 1 + (uint32_t)std::ceil((float)(inputH - windowHeight) / (float)poolStrideH);

    int64_t tmpValue = 0;
    int64_t value = 0;
    for (uint32_t OD = 0; OD < numFilters; OD++)
    {
        for (uint32_t POW = 0; POW < poolOutW; POW++)
        {
            uint32_t inIdxW = numFilters * POW * poolStrideW;
            for (uint32_t POH = 0; POH < poolOutH; POH++)
            {
                uint32_t inIdxH = numFilters * inputW * POH * poolStrideH;
                tmpValue = 0;

                for (uint32_t OW = 0; OW < windowWidth; OW++)
                {
                    uint32_t winIdxW = numFilters * OW;

                    for (uint32_t OH = 0; OH < windowHeight; OH++) {

                        uint32_t winIdxH = numFilters * inputW * OH;

                        if (config->RequestConfig->Transform.Pooling.pooling.type == INTEL_MAX_POOLING)
                        {
                            if ((POW * poolStrideW <= (inputW - 1)) && (POH * poolStrideH <= (inputH - 1)))
                            {
                                tmpValue = I[OD + inIdxW + inIdxH];
                            }

                            if ((POW * poolStrideW + OW <= (inputW - 1)) && (POH * poolStrideH + OH <= (inputH - 1)))
                            {
                                tmpValue = I[OD + inIdxW + inIdxH + winIdxW + winIdxH];
                            }

                            if ((OW == 0) && (OH == 0))
                            {
                                value = tmpValue;
                            }
                            else if (value < tmpValue)
                            {
                                value = tmpValue;
                            }
                        }
                        else if (config->RequestConfig->Transform.Pooling.pooling.type == INTEL_SUM_POOLING)
                        {

                            if ((POW * poolStrideW + OW <= (inputW - 1)) && (POH * poolStrideH + OH <= (inputH - 1)))
                            {
                                tmpValue += I[OD + inIdxW + inIdxH + winIdxW + winIdxH];
                            }

                            if (value > 32767)
                            {
                                value = 32767;
                            }
                            else if (value < -32768)
                            {
                                value = -32768;
                            }
                            else
                            {
                                value = tmpValue;
                            }
                        }

                    }
                }

                O[POH * poolOutW * numFilters + POW * numFilters + OD] = (int32_t)value;
            }
        }
    }

}
void Convolution2DKernelImpl1B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    uint32_t inputDepth = config->RequestConfig->Transform.InputDimensions.depth;
    uint32_t inputHeight = config->RequestConfig->Transform.InputDimensions.height;
    uint32_t inputWidth = config->RequestConfig->Transform.InputDimensions.width;

    uint32_t numFilters = config->RequestConfig->Transform.Convolution.filters.count;
    uint32_t filterHeight = config->RequestConfig->Transform.Convolution.filters.dimensions.height;
    uint32_t filterWidth = config->RequestConfig->Transform.Convolution.filters.dimensions.width;
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth);
    uint32_t filterPadding = (ALIGN(memForFilter, 16) - memForFilter);

    uint32_t padHeight = config->RequestConfig->Transform.Convolution.zeroPadding.height;
    uint32_t padWidth = config->RequestConfig->Transform.Convolution.zeroPadding.width;

    uint32_t strideHeight = config->RequestConfig->Transform.Convolution.stride.height;
    uint32_t strideWidth = config->RequestConfig->Transform.Convolution.stride.width;

    auto biasMode = config->RequestConfig->Transform.Convolution.biases.mode;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t inWidthMax = inputWidth + padWidth - 1;
    uint32_t inHeightMax = inputHeight + padHeight - 1;

    const int8_t* const I = (int8_t*)config->RequestConfig->Inputs;
    int32_t* O = (int32_t*)config->RequestConfig->Outputs;
    int8_t* F = (int8_t*)config->RequestConfig->Transform.Convolution.filters.filtersData;

    auto biasPrecission = config->RequestConfig->Transform.Convolution.biases.dataMode;
    void* biasData = config->RequestConfig->Transform.Convolution.biases.biasesData;

    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    for (uint32_t OD = 0; OD < numFilters; OD++) { //Output depth or #filters

        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

        for (uint32_t OW = 0; OW < outWidth; OW++) { //Output width
            for (uint32_t OH = 0; OH < outHeight; OH++) {    //Output height

                int64_t outVal;// = &O[OH * outWidth * numFilters + OW * numFilters + OD]; //NHWC order
                if (biasMode == GNA_BIAS_PER_KERNEL) {
                    outVal = getBias(biasData, OD, biasPrecission);
                }
                else if (biasMode == GNA_BIAS_NOT_SUPPORTED) {
                    outVal = 0;
                }
                else {
                    outVal = getBias(biasData, numFilters*outWidth*OH + numFilters * OW + OD, biasPrecission);
                }


                for (uint32_t z = 0; z < inputDepth; z++) {   //Input depth
                    for (uint32_t w = 0; w < filterWidth; w++) { //input height

                        uint32_t wIdx = OW * strideWidth + w;
                        uint32_t fIdxW = inputDepth * w;
                        uint32_t inIdxW = (((OW*strideWidth) + w - padWidth) * (inputDepth));

                        for (uint32_t h = 0; h < filterHeight; h++) { //input width

                            uint32_t inIdxH = (((OH*strideHeight) + h - padHeight) * (inputDepth*inputWidth));
                            uint32_t fIdxH = inputDepth * filterWidth * h;
                            uint32_t hIdx = OH * strideHeight + h;

                            if (wIdx < padWidth || wIdx > inWidthMax || hIdx < padHeight || hIdx > inHeightMax) { //padding
                                continue;
                            }
                            outVal += (int64_t)(I[inIdxH + inIdxW + z]) * F[fIdxN + fIdxH + fIdxW + z];
                        }
                    }
                }

                O[OH * outWidth * numFilters + OW * numFilters + OD] = (int32_t)outVal;
            }
        }
    }
}
void Convolution2DKernelImpl1B2B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    uint32_t inputDepth = config->RequestConfig->Transform.InputDimensions.depth;
    uint32_t inputHeight = config->RequestConfig->Transform.InputDimensions.height;
    uint32_t inputWidth = config->RequestConfig->Transform.InputDimensions.width;

    uint32_t numFilters = config->RequestConfig->Transform.Convolution.filters.count;
    uint32_t filterHeight = config->RequestConfig->Transform.Convolution.filters.dimensions.height;
    uint32_t filterWidth = config->RequestConfig->Transform.Convolution.filters.dimensions.width;
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth);
    uint32_t filterPadding = (ALIGN(memForFilter, 16) - memForFilter);

    uint32_t padHeight = config->RequestConfig->Transform.Convolution.zeroPadding.height;
    uint32_t padWidth = config->RequestConfig->Transform.Convolution.zeroPadding.width;

    uint32_t strideHeight = config->RequestConfig->Transform.Convolution.stride.height;
    uint32_t strideWidth = config->RequestConfig->Transform.Convolution.stride.width;

    auto biasMode = config->RequestConfig->Transform.Convolution.biases.mode;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t inWidthMax = inputWidth + padWidth - 1;
    uint32_t inHeightMax = inputHeight + padHeight - 1;

    const int16_t* const I = (int16_t*)config->RequestConfig->Inputs;
    int32_t* O = (int32_t*)config->RequestConfig->Outputs;
    int8_t* F = (int8_t*)config->RequestConfig->Transform.Convolution.filters.filtersData;

    auto biasPrecission = config->RequestConfig->Transform.Convolution.biases.dataMode;
    void* biasData = config->RequestConfig->Transform.Convolution.biases.biasesData;

    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    for (uint32_t OD = 0; OD < numFilters; OD++) { //Output depth or #filters

        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

        for (uint32_t OW = 0; OW < outWidth; OW++) { //Output width
            for (uint32_t OH = 0; OH < outHeight; OH++) {    //Output height

                int64_t outVal;// = &O[OH * outWidth * numFilters + OW * numFilters + OD]; //NHWC order
                if (biasMode == GNA_BIAS_PER_KERNEL) {
                    outVal = getBias(biasData, OD, biasPrecission);
                }
                else if (biasMode == GNA_BIAS_NOT_SUPPORTED) {
                    outVal = 0;
                }
                else {
                    outVal = getBias(biasData, numFilters*outWidth*OH + numFilters * OW + OD, biasPrecission);
                }


                for (uint32_t z = 0; z < inputDepth; z++) {   //Input depth
                    for (uint32_t w = 0; w < filterWidth; w++) { //input height

                        uint32_t wIdx = OW * strideWidth + w;
                        uint32_t fIdxW = inputDepth * w;
                        uint32_t inIdxW = (((OW*strideWidth) + w - padWidth) * (inputDepth));

                        for (uint32_t h = 0; h < filterHeight; h++) { //input width

                            uint32_t inIdxH = (((OH*strideHeight) + h - padHeight) * (inputDepth*inputWidth));
                            uint32_t fIdxH = inputDepth * filterWidth * h;
                            uint32_t hIdx = OH * strideHeight + h;

                            if (wIdx < padWidth || wIdx > inWidthMax || hIdx < padHeight || hIdx > inHeightMax) { //padding
                                continue;
                            }
                            outVal += (int64_t)(I[inIdxH + inIdxW + z]) * F[fIdxN + fIdxH + fIdxW + z];
                        }
                    }
                }

                O[OH * outWidth * numFilters + OW * numFilters + OD] = (int32_t)outVal;
            }
        }
    }
}
void Convolution2DKernelImpl2B1B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    uint32_t inputDepth = config->RequestConfig->Transform.InputDimensions.depth;
    uint32_t inputHeight = config->RequestConfig->Transform.InputDimensions.height;
    uint32_t inputWidth = config->RequestConfig->Transform.InputDimensions.width;

    uint32_t numFilters = config->RequestConfig->Transform.Convolution.filters.count;
    uint32_t filterHeight = config->RequestConfig->Transform.Convolution.filters.dimensions.height;
    uint32_t filterWidth = config->RequestConfig->Transform.Convolution.filters.dimensions.width;
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth * 2);
    uint32_t filterPadding = (ALIGN(memForFilter, 16) - memForFilter) / 2;

    uint32_t padHeight = config->RequestConfig->Transform.Convolution.zeroPadding.height;
    uint32_t padWidth = config->RequestConfig->Transform.Convolution.zeroPadding.width;

    uint32_t strideHeight = config->RequestConfig->Transform.Convolution.stride.height;
    uint32_t strideWidth = config->RequestConfig->Transform.Convolution.stride.width;

    auto biasMode = config->RequestConfig->Transform.Convolution.biases.mode;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t inWidthMax = inputWidth + padWidth - 1;
    uint32_t inHeightMax = inputHeight + padHeight - 1;

    const int8_t* const I = (int8_t*)config->RequestConfig->Inputs;
    int32_t* O = (int32_t*)config->RequestConfig->Outputs;
    int16_t* F = (int16_t*)config->RequestConfig->Transform.Convolution.filters.filtersData;

    auto biasPrecission = config->RequestConfig->Transform.Convolution.biases.dataMode;
    void* biasData = config->RequestConfig->Transform.Convolution.biases.biasesData;

    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    for (uint32_t OD = 0; OD < numFilters; OD++) { //Output depth or #filters

        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

        for (uint32_t OW = 0; OW < outWidth; OW++) { //Output width
            for (uint32_t OH = 0; OH < outHeight; OH++) {    //Output height

                int64_t outVal;// = &O[OH * outWidth * numFilters + OW * numFilters + OD]; //NHWC order
                if (biasMode == GNA_BIAS_PER_KERNEL) {
                    outVal = getBias(biasData, OD, biasPrecission);
                }
                else if (biasMode == GNA_BIAS_NOT_SUPPORTED) {
                    outVal = 0;
                }
                else {
                    outVal = getBias(biasData, numFilters*outWidth*OH + numFilters * OW + OD, biasPrecission);
                }


                for (uint32_t z = 0; z < inputDepth; z++) {   //Input depth
                    for (uint32_t w = 0; w < filterWidth; w++) { //input height

                        uint32_t wIdx = OW * strideWidth + w;
                        uint32_t fIdxW = inputDepth * w;
                        uint32_t inIdxW = (((OW*strideWidth) + w - padWidth) * (inputDepth));

                        for (uint32_t h = 0; h < filterHeight; h++) { //input width

                            uint32_t inIdxH = (((OH*strideHeight) + h - padHeight) * (inputDepth*inputWidth));
                            uint32_t fIdxH = inputDepth * filterWidth * h;
                            uint32_t hIdx = OH * strideHeight + h;

                            if (wIdx < padWidth || wIdx > inWidthMax || hIdx < padHeight || hIdx > inHeightMax) { //padding
                                continue;
                            }
                            outVal += (int64_t)(I[inIdxH + inIdxW + z]) * F[fIdxN + fIdxH + fIdxW + z];
                        }
                    }
                }

                O[OH * outWidth * numFilters + OW * numFilters + OD] = (int32_t)outVal;
            }
        }
    }
}
void Convolution2DKernelImpl2B2B(ExecutionKernelConfig<ConvolutionConfig2D> const * const config)
{
    uint32_t inputDepth = config->RequestConfig->Transform.InputDimensions.depth;
    uint32_t inputHeight = config->RequestConfig->Transform.InputDimensions.height;
    uint32_t inputWidth = config->RequestConfig->Transform.InputDimensions.width;

    uint32_t numFilters = config->RequestConfig->Transform.Convolution.filters.count;
    uint32_t filterHeight = config->RequestConfig->Transform.Convolution.filters.dimensions.height;
    uint32_t filterWidth = config->RequestConfig->Transform.Convolution.filters.dimensions.width;
    uint32_t memForFilter = (filterHeight * filterWidth * inputDepth * 2);
    uint32_t filterPadding = (ALIGN(memForFilter, 16) - memForFilter) /2;

    uint32_t padHeight = config->RequestConfig->Transform.Convolution.zeroPadding.height;
    uint32_t padWidth = config->RequestConfig->Transform.Convolution.zeroPadding.width;

    uint32_t strideHeight = config->RequestConfig->Transform.Convolution.stride.height;
    uint32_t strideWidth = config->RequestConfig->Transform.Convolution.stride.width;

    auto biasMode = config->RequestConfig->Transform.Convolution.biases.mode;

    uint32_t inputHeightWPad = inputHeight + 2 * padHeight;
    uint32_t inputWidthWPad = inputWidth + 2 * padWidth;
    uint32_t inWidthMax = inputWidth + padWidth - 1;
    uint32_t inHeightMax = inputHeight + padHeight - 1;

    const int16_t* const I = (int16_t*)config->RequestConfig->Inputs;
    int32_t* O = (int32_t*)config->RequestConfig->Outputs;
    int16_t* F = (int16_t*)config->RequestConfig->Transform.Convolution.filters.filtersData;

    auto biasPrecission = config->RequestConfig->Transform.Convolution.biases.dataMode;
    void* biasData = config->RequestConfig->Transform.Convolution.biases.biasesData;

    uint32_t outWidth = 1 + ((inputWidthWPad - filterWidth) / strideWidth);
    uint32_t outHeight = 1 + ((inputHeightWPad - filterHeight) / strideHeight);

    for (uint32_t OD = 0; OD < numFilters; OD++) { // Output depth

        uint32_t fIdxN = (OD * (inputDepth * filterWidth * filterHeight + filterPadding));

        for (uint32_t OW = 0; OW < outWidth; OW++) {
            for (uint32_t OH = 0; OH < outHeight; OH++) {

                int64_t outVal;// = &O[OH * outWidth * numFilters + OW * numFilters + OD]; //NHWC order
                if (biasMode == GNA_BIAS_PER_KERNEL) {
                    outVal = getBias(biasData, OD, biasPrecission);
                }
                else if (biasMode == GNA_BIAS_NOT_SUPPORTED) {
                    outVal = 0;
                }
                else {
                    outVal = getBias(biasData, numFilters*outWidth*OH + numFilters * OW + OD, biasPrecission);
                }


                for (uint32_t z = 0; z < inputDepth; z++) {
                    for (uint32_t w = 0; w < filterWidth; w++) { //input height

                        uint32_t wIdx = OW * strideWidth + w;
                        uint32_t fIdxW = inputDepth * w;
                        uint32_t inIdxW = (((OW*strideWidth) + w - padWidth) * (inputDepth));

                        for (uint32_t h = 0; h < filterHeight; h++) {

                            uint32_t inIdxH = (((OH*strideHeight) + h - padHeight) * (inputDepth*inputWidth));
                            uint32_t fIdxH = inputDepth * filterWidth * h;
                            uint32_t hIdx = OH * strideHeight + h;

                            if (wIdx < padWidth || wIdx > inWidthMax || hIdx < padHeight || hIdx > inHeightMax) { //padding
                                continue;
                            }
                            outVal += (int64_t)(I[inIdxH + inIdxW + z]) * F[fIdxN + fIdxH + fIdxW + z];
                        }
                    }
                }

                O[OH * outWidth * numFilters + OW * numFilters + OD] = (int32_t)outVal;
            }
        }
    }
}
