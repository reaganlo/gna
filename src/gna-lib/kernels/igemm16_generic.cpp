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

#include "igemv16.h"

#include "KernelArguments.h"

#include <cstdint>

void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount *
                                           config->RequestConfig->Transform.bytesPerBias);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        bias += config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j;
    uint32_t k;
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias + (config->RequestConfig->Transform.outputElementCount *
                                           config->RequestConfig->Transform.bytesPerBias);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        bias += config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j;
    uint32_t k;
    int8_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;

    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;
    int8_t const * const biasEnd = bias
        + (config->RequestConfig->Transform.outputElementCount
                * config->RequestConfig->Transform.bytesPerBias);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    for (; bias < biasEnd;)
    {
        input = (int8_t*)config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(bias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        bias += config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int8_t const * multiBias = (int8_t*)config->RequestConfig->Transform.multiBias;
    int8_t const * const biasEnd = multiBias +
        (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        multiBias += config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineMultiBiasKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int16_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int8_t const * multiBias = static_cast<int8_t const *>(config->RequestConfig->Transform.multiBias);
    int8_t const * const biasEnd = multiBias +
        (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        multiBias += config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    }
}

void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t j, k;
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * input;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    int8_t const * multiBias = (int8_t*)config->RequestConfig->Transform.multiBias;
    int8_t const * const biasEnd = multiBias +
        (config->RequestConfig->Transform.outputElementCount * config->RequestConfig->Transform.bytesPerBias * config->RequestConfig->Transform.multiBiasVectorCount);

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    for (; multiBias < biasEnd;)
    {
        input = (int8_t*)config->Intermediate->d0;
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            *output = getBias(multiBias, config->RequestConfig->Transform.bytesPerBias);

            for (k = 0; k < config->RequestConfig->Transform.inputElementCount; k++)
            {
                *output += weight[k] * *input++;
            }

            output++;
        }
        weight += config->RequestConfig->Transform.inputElementCount;
        multiBias += config->RequestConfig->Transform.multiBiasVectorCount * config->RequestConfig->Transform.bytesPerBias;
    }
}
