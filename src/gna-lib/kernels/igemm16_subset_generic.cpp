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

void AffineActiveListKernelImpl2B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t i, j, k, l;
    int16_t const * input;
    int16_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, 
                                                       config->input, config->fvBuffers->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (l = 0; l < al->count; l++)
    {
        i = al->indices[l];

        input = config->fvBuffers->d0;
        weight = config->weights2B + i*config->inputElementCount;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            config->output[l*config->inputVectorCount + j] = config->biasesSimple[i];
            for (k = 0; k < config->inputElementCount; k++)
            {
                config->output[l*config->inputVectorCount + j] += weight[k] * *input++;
            }
        }
    }
}

void AffineMultiBiasActiveListKernelImpl2B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t i, j, k, l;
    int16_t const * input;
    int16_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, 
                                                       config->input, config->fvBuffers->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (l = 0; l < al->count; l++)
    {
        i = al->indices[l];

        input = config->fvBuffers->d0;
        weight = config->weights2B + i*config->inputElementCount;
        for (j = 0; j < config->inputVectorCount; j++)
        {
            config->output[l*config->inputVectorCount + j] = config->multiBias[i*config->multiBiasVectorCount];
            for (k = 0; k < config->inputElementCount; k++)
            {
                config->output[l*config->inputVectorCount + j] += weight[k] * *input++;
            }
        }
    }
}
