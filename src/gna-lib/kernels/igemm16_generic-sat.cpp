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

#include "KernelArguments.h"

#include "common.h"
#include "gna-api-types-xnn.h"

void AffineKernelImpl2B(AffineConfig const * const config)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    int16_t const * input;
    int16_t const * weight;

    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl(&transposeConfig);

    int64_t sum = 0;
    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(config->biasesSimple, i, config->bytesPerBias);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk * kpartial;
                weight = config->weights2B + i*config->inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[i*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = (int64_t)config->output[i*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineKernelImpl2B2B(AffineConfig const * const config)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    int16_t const * input;
    int16_t const * weight;

    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    int64_t sum = 0;
    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(config->biasesSimple, i, config->bytesPerBias);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk * kpartial;
                weight = config->weights2B + i*config->inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[i*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = (int64_t)config->output[i*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineKernelImpl2B1B(AffineConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;
    int8_t const * input;
    int16_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl1B(&transposeConfig);

    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    int64_t sum = 0;
    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(config->biasesSimple, i, config->bytesPerBias);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = ((int8_t*)config->execution->Intermediate->d0) + j*config->inputElementCount + kk * kpartial;
                weight = config->weights2B + i*config->inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[i*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = (int64_t)config->output[i*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineMultiBiasKernelImpl2B(AffineConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    const uint32_t kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    const uint32_t nKpartial = config->inputElementCount / kpartial;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl(&transposeConfig);

    int16_t const * input;
    int16_t const * weight;

    int64_t sum = 0;
    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(config->multiBias, i*config->multiBiasVectorCount, config->bytesPerBias);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk*kpartial;
                weight = config->weights2B + i*config->inputElementCount + kk*kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++)
                {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[i*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = config->output[i*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineMultiBiasKernelImpl2B2B(AffineConfig const * const config)
{
    int64_t sum = 0;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    const uint32_t kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    const uint32_t nKpartial = config->inputElementCount / kpartial;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    int16_t const * input;
    int16_t const * weight;

    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(config->multiBias, i*config->multiBiasVectorCount, config->bytesPerBias);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk*kpartial;
                weight = config->weights2B + i*config->inputElementCount + kk*kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++)
                {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[i*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = config->output[i*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}

void AffineMultiBiasKernelImpl2B1B(AffineConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t kk;
    const uint32_t kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1]) / config->inputVectorCount;
    const uint32_t nKpartial = config->inputElementCount / kpartial;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl1B(&transposeConfig);

    int8_t const * input;
    int16_t const * weight;

    int64_t sum = 0;
    for (i = 0; i < config->outputElementCount; i++)
    {
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum = getBias(config->multiBias, i*config->multiBiasVectorCount, config->bytesPerBias);

            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                input = ((int8_t*)config->execution->Intermediate->d0) + j*config->inputElementCount + kk*kpartial;
                weight = config->weights2B + i*config->inputElementCount + kk*kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++)
                {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[i*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = config->output[i*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}
