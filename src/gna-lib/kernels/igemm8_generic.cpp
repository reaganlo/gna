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
#include "igemv16.h"

#include "KernelArguments.h"

#include "common.h"

#include <cstdint>

void AffineKernelImpl1B(AffineConfig const * const config)
{
    uint32_t j;
    uint32_t k;
    int8_t const * weight = config->weights1B;
    int16_t const * input;
    int32_t * output = config->output;
    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            *output = 0;
            for (k = 0; k < config->inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }
            *output *= bias->multiplier;
            *output++ += bias->bias;
        }
        weight += config->inputElementCount;
        bias++;
    }
}
void AffineKernelImpl1B1B(AffineConfig const * const config)
{
    uint32_t j;
    uint32_t k;
    int8_t const * input;
    int8_t const * weight = config->weights1B;

    int32_t * output = config->output;
    int8_t const * bias = (int8_t*)config->biasesSimple;
    int8_t const * const biasEnd = bias + (config->outputElementCount * config->bytesPerBias);

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl1B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = (int8_t*)config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            *output = getBias(bias, config->bytesPerBias);

            for (k = 0; k < config->inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->inputElementCount;
        bias += config->bytesPerBias;
    }
}
void AffineKernelImpl1B2B(AffineConfig const * const config)
{
    uint32_t j, k;
    int8_t const * weight = config->weights1B;
    int16_t const * input;
    int32_t * output = config->output;
    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            *output = 0;
            for (k = 0; k < config->inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }
            *output *= bias->multiplier;
            *output++ += bias->bias;
        }
        weight += config->inputElementCount;
        bias++;
    }
}

void AffineMultiBiasKernelImpl1B(AffineConfig const * const config)
{
    uint32_t j;
    uint32_t k;
    int16_t const * input;
    int32_t * output = config->output;
    nn_scaling const * const biasEnd = config->weightScaleFactors + config->outputElementCount;
    int8_t const * weight = config->weights1B;
    nn_scaling const * weightScaleFactors = config->weightScaleFactors;
    int8_t const * multiBias = (int8_t*)config->multiBias;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (; weightScaleFactors < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; ++j)
        {
            *output = 0;
            for (k = 0; k < config->inputElementCount; ++k)
            {
                *output += weight[k] * *input++;
            }
            *output *= weightScaleFactors->multiplier;

            *output++ += getBias(multiBias, config->bytesPerBias);
        }
        weight += config->inputElementCount;
        multiBias += config->multiBiasVectorCount * config->bytesPerBias;
        weightScaleFactors++;
    }
}

void AffineMultiBiasKernelImpl1B2B(AffineConfig const * const config)
{
    uint32_t j, k;
    int16_t const * input;
    int32_t * output = config->output;
    nn_scaling const * const biasEnd = config->weightScaleFactors + config->outputElementCount;
    int8_t const * weight = config->weights1B;
    nn_scaling const * weightScaleFactors = config->weightScaleFactors;
    int8_t const * multiBias = (int8_t*)config->multiBias;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    for (; weightScaleFactors < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; ++j)
        {
            *output = 0;
            for (k = 0; k < config->inputElementCount; ++k)
            {
                *output += weight[k] * *input++;
            }
            *output *= weightScaleFactors->multiplier;

            *output++ += getBias(multiBias, config->bytesPerBias);
        }
        weight += config->inputElementCount;
        multiBias += config->multiBiasVectorCount * config->bytesPerBias;
        weightScaleFactors++;
    }
}

void AffineMultiBiasKernelImpl1B1B(AffineConfig const * const config)
{
    uint32_t j, k;
    int32_t * output = config->output;
    int8_t const * input;
    int8_t const * weight = config->weights1B;
    int8_t const * multiBias = (int8_t*)config->multiBias;
    int8_t const * const biasEnd = multiBias +
        (config->outputElementCount * config->bytesPerBias * config->multiBiasVectorCount);

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl1B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = (int8_t*)config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->bytesPerBias);

            for (k = 0; k < config->inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->inputElementCount;
        multiBias += config->multiBiasVectorCount * config->bytesPerBias;
    }
}
