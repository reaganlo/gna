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
#include "igemv16.h"

void AffineKernelImpl2B(AffineConfig const * const config)
{
    uint32_t j, k;
    int16_t const * input;
    int16_t const * weight = config->weights2B;

    int32_t * output = config->output;
    int8_t const * bias = (int8_t*)config->biasesSimple;
    int8_t const * const biasEnd = bias + (config->outputElementCount * config->bytesPerBias);

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            if (config->bytesPerBias == 1)
                *output = *bias;
            else if (config->bytesPerBias == 2)
                *output = *(int16_t*)bias;
            else if (config->bytesPerBias == 4)
                *output = *(int32_t*)bias;

            for (k = 0; k < config->inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->inputElementCount;
        bias+= config->bytesPerBias;
    }
}

void AffineKernelImpl2B2B(AffineConfig const * const config)
{
    uint32_t j;
    uint32_t k;
    int16_t const * input;
    int16_t const * weight = config->weights2B;

    int32_t * output = config->output;
    int8_t const * bias = (int8_t*)config->biasesSimple;
    int8_t const * const biasEnd = bias + (config->outputElementCount * config->bytesPerBias);

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            if (config->bytesPerBias == 1)
                *output = *bias;
            else if (config->bytesPerBias == 2)
                *output = *(int16_t*)bias;
            else if (config->bytesPerBias == 4)
                *output = *(int32_t*)bias;

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


void AffineKernelImpl2B1B(AffineConfig const * const config)
{
    uint32_t j;
    uint32_t k;
    int8_t const * input;
    int16_t const * weight = config->weights2B;

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
            if (config->bytesPerBias == 1)
                *output = *bias;
            else if (config->bytesPerBias == 2)
                *output = *(int16_t*)bias;
            else if (config->bytesPerBias == 4)
                *output = *(int32_t*)bias;

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

void AffineMultiBiasKernelImpl2B(AffineConfig const * const config)
{
    uint32_t j, k;
    int32_t * output = config->output;
    int16_t const * input;
    int16_t const * weight = config->weights2B;
    int8_t const * multiBias = (int8_t*)config->multiBias;
    int8_t const * const biasEnd = multiBias + (config->outputElementCount * config->bytesPerBias)*config->multiBiasVectorCount;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            if (config->bytesPerBias == 1)
                *output = *multiBias;
            else if (config->bytesPerBias == 2)
                *output = *(int16_t*)multiBias;
            else if (config->bytesPerBias == 4)
                *output = *(int32_t*)multiBias;

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

void AffineMultiBiasKernelImpl2B2B(AffineConfig const * const config)
{
    uint32_t j, k;
    int32_t * output = config->output;
    int16_t const * input;
    int16_t const * weight = config->weights2B;
    int8_t const * multiBias = (int8_t*)config->multiBias;
    int8_t const * const biasEnd = multiBias + (config->outputElementCount * config->bytesPerBias)*config->multiBiasVectorCount;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            if (config->bytesPerBias == 1)
                *output = *multiBias;
            else if (config->bytesPerBias == 2)
                *output = *(int16_t*)multiBias;
            else if (config->bytesPerBias == 4)
                *output = *(int32_t*)multiBias;

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

void AffineMultiBiasKernelImpl2B1B(AffineConfig const * const config)
{
    uint32_t j, k;
    int32_t * output = config->output;
    int8_t const * input;
    int16_t const * weight = config->weights2B;
    int8_t const * multiBias = (int8_t*)config->multiBias;
    int8_t const * const biasEnd = multiBias + (config->outputElementCount * config->bytesPerBias)*config->multiBiasVectorCount;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl1B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = (int8_t*)config->execution->Intermediate->d0;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            if (config->bytesPerBias == 1)
                *output = *multiBias;
            else if (config->bytesPerBias == 2)
                *output = *(int16_t*)multiBias;
            else if (config->bytesPerBias == 4)
                *output = *(int32_t*)multiBias;

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
