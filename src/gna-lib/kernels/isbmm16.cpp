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
#include "igemv.h"

#include "KernelArguments.h"

#include <cstdint>

void DiagonalKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    auto const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig->Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias, i)
                + (weight[i] * input[i * config->RequestConfig->Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig->Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}

void DiagonalKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    auto const * input = reinterpret_cast<int16_t const *>(config->RequestConfig->Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig->Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias, i)
                + (weight[i] * input[i * config->RequestConfig->Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig->Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}

void DiagonalKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->RequestConfig->Transform.weights2B;
    auto const * input = reinterpret_cast<int8_t const *>(config->RequestConfig->Inputs);
    int32_t * output = reinterpret_cast<int32_t *>(config->RequestConfig->Outputs);
    int8_t const * bias = (int8_t*)config->RequestConfig->Transform.biasesSimple;

    for (i = 0; i < config->RequestConfig->Transform.outputElementCount; i++)
    {
        for (j = 0; j < config->RequestConfig->Transform.inputVectorCount; j++)
        {
            sum = getBias(bias, config->RequestConfig->Transform.bytesPerBias, i)
                + (weight[i] * input[i * config->RequestConfig->Transform.inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->RequestConfig->Transform.inputVectorCount + j], config->SaturationCount);
        }
    }
}
