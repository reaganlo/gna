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

void DiagonalKernelImpl2B(AffineConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->weights2B;
    int16_t const * input = config->input;
    int32_t * output = config->output;
    int8_t const * bias = (int8_t*)config->biasesSimple;

    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(bias, i, config->bytesPerBias)
                + (weight[i] * input[i * config->inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->inputVectorCount + j], config->execution->SaturationCount);
        }
    }
}

void DiagonalKernelImpl2B2B(AffineConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->weights2B;
    int16_t const * input = config->input;
    int32_t * output = config->output;
    int8_t const * bias = (int8_t*)config->biasesSimple;

    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(bias, i, config->bytesPerBias)
                + (weight[i] * input[i * config->inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->inputVectorCount + j], config->execution->SaturationCount);
        }
    }
}

void DiagonalKernelImpl2B1B(AffineConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    int16_t const * weight = config->weights2B;
    int8_t const * input =(int8_t*) config->input;
    int32_t * output = config->output;
    int8_t const * bias = (int8_t*)config->biasesSimple;

    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(bias, i, config->bytesPerBias)
                + (weight[i] * input[i * config->inputVectorCount + j]);

            saturate_store_out(&sum, &output[i * config->inputVectorCount + j], config->execution->SaturationCount);
        }
    }
}
