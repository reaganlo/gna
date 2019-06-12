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
#include "igemv8.h"
#include "igemv16.h"

#include "KernelArguments.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>

void AffineActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    uint32_t m;
    int64_t sum;
    int32_t acc;
    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    int16_t const * input;
    int8_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    for (l = 0; l < al->count; l++) {
        i = al->indices[l];
        for (j = 0; j < config->inputVectorCount; j++) {
            sum = config->biasesCompound[i].bias;
            for (kk = 0; kk < nKpartial + 1; kk++) {
                niters = kpartial < config->inputElementCount - kk * kpartial ? kpartial : config->inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; k++)
                {
                    input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk * kpartial + k * 512;
                    weight = config->weights1B + i*config->inputElementCount + kk * kpartial + k * 512;
                    for (m = 0; m < 512; m++)
                    {
                        acc += weight[m] * input[m];
                    }
                    sum += acc * config->biasesCompound[i].multiplier;
                    acc = 0;
                }

                input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->weights1B + i*config->inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; k++)
                {
                    acc += weight[k] * input[k];
                }
                sum += acc * config->biasesCompound[i].multiplier;

                saturate_store_out(&sum, &config->output[l*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = (int64_t)config->output[l*config->inputVectorCount + j];
            }
        }
    }
}

void AffineActiveListKernelImpl1B2B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t acc_iters;
    uint32_t rem_iters;
    uint32_t niters;
    uint32_t nKpartial;
    uint32_t kpartial;
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    uint32_t m;
    int64_t sum;
    int32_t acc;
    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1 + XNN_N_GROUP_MAX]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    int16_t const * input;
    int8_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl2B(&transposeConfig);

    for (l = 0; l < al->count; l++) {
        i = al->indices[l];
        for (j = 0; j < config->inputVectorCount; j++) {
            sum = config->biasesCompound[i].bias;
            for (kk = 0; kk < nKpartial + 1; kk++) {
                niters = kpartial < config->inputElementCount - kk * kpartial ? kpartial : config->inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; k++)
                {
                    input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk * kpartial + k * 512;
                    weight = config->weights1B + i*config->inputElementCount + kk * kpartial + k * 512;
                    for (m = 0; m < 512; m++)
                    {
                        acc += weight[m] * input[m];
                    }
                    sum += acc * config->biasesCompound[i].multiplier;
                    acc = 0;
                }

                input = config->execution->Intermediate->d0 + j*config->inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->weights1B + i*config->inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; k++)
                {
                    acc += weight[k] * input[k];
                }
                sum += acc * config->biasesCompound[i].multiplier;

                saturate_store_out(&sum, &config->output[l*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = (int64_t)config->output[l*config->inputVectorCount + j];
            }
        }
    }
}

void AffineActiveListKernelImpl1B1B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
    uint32_t l;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;
    kpartial = (config->execution->BufferElementCount[config->inputVectorCount - 1]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount,
        config->input, config->execution->Intermediate->d0 };
    TransposeKernelImpl1B(&transposeConfig);

    int8_t const * input;
    int8_t const * weight;

    int64_t sum = 0;
    for (l = 0; l < al->count; l++) {
        i = al->indices[l];
        for (j = 0; j < config->inputVectorCount; j++) {

            sum = getBias(config->biasesSimple, config->bytesPerBias, i);

            for (kk = 0; kk < nKpartial + 1; kk++) {
                input = ((int8_t*)config->execution->Intermediate->d0) + j*config->inputElementCount + kk * kpartial;
                weight = config->weights1B + i*config->inputElementCount + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < config->inputElementCount); k++) {
                    sum += weight[k] * input[k];
                }
                saturate_store_out(&sum, &config->output[l*config->inputVectorCount + j], config->execution->SaturationCount);
                sum = (int64_t)config->output[l*config->inputVectorCount + j]; // load the temp sum
            }
        }
    }
}
