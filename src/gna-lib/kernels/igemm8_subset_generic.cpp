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

void AffineActiveListKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    auto inputVectorCount = config->RequestConfig->Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig->Transform.inputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl(&transposeConfig);

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        input = config->Intermediate->d0;
        weight = config->RequestConfig->Transform.weights1B+i*inputElementCount;
        for (j = 0; j < inputVectorCount; j++)
        {
            output[l*inputVectorCount + j] = 0;
            for (k = 0; k < inputElementCount; k++)
            {
                output[l*inputVectorCount + j] += weight[k] * *input++;
            }
            output[l*inputVectorCount + j] *= config->RequestConfig->Transform.biasesCompound[i].multiplier;
            output[l*inputVectorCount + j] += config->RequestConfig->Transform.biasesCompound[i].bias;
        }
    }
}

void AffineActiveListKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i, j, k, l;
    int16_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    auto inputVectorCount = config->RequestConfig->Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig->Transform.inputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl2B(&transposeConfig);

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];
        input = config->Intermediate->d0;
        weight = config->RequestConfig->Transform.weights1B + i*inputElementCount;
        for (j = 0; j < inputVectorCount; j++)
        {
            output[l*inputVectorCount + j] = 0;
            for (k = 0; k < inputElementCount; k++)
            {
                output[l*inputVectorCount + j] += weight[k] * *input++;
            }
            output[l*inputVectorCount + j] *= config->RequestConfig->Transform.biasesCompound[i].multiplier;
            output[l*inputVectorCount + j] += config->RequestConfig->Transform.biasesCompound[i].bias;
        }
    }
}
void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al)
{
    uint32_t i, j, k, l;
    int8_t const * input;
    int8_t const * weight;
    auto *output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);

    auto inputVectorCount = config->RequestConfig->Transform.inputVectorCount;
    auto inputElementCount = config->RequestConfig->Transform.inputElementCount;

    auto transposeConfig = TransposeConfig::MakeFrom(config);
    TransposeKernelImpl1B(&transposeConfig);

    for (l = 0; l < al.count; l++)
    {
        i = al.indices[l];

        input = (int8_t*)config->Intermediate->d0;
        weight = config->RequestConfig->Transform.weights1B + i*inputElementCount;
        for (j = 0; j < inputVectorCount; j++)
        {
            auto bias = getBias((void*)config->RequestConfig->Transform.biasesSimple, config->RequestConfig->Transform.bytesPerBias, i);
            output[l*config->RequestConfig->Transform.inputVectorCount + j] = bias;

            for (k = 0; k < inputElementCount; k++)
            {
                output[l*inputVectorCount + j] += weight[k] * *input++;
            }
        }
    }
}
