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

void AffineActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t niters, acc_iters, rem_iters;
    uint32_t i,j,k,l,m;
    int64_t sum;
    int32_t acc;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;
    kpartial = (hw_buf_size[config->inputVectorCount - 1]) / config->inputVectorCount;
    nKpartial = config->inputElementCount / kpartial;

    int16_t const * input;
    int8_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, 
                                                       config->input, config->fvBuffers->d0 };
    TransposeKernelImpl(&transposeConfig);

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
                    input = config->fvBuffers->d0 + j*config->inputElementCount + kk * kpartial + k * 512;
                    weight = config->weights1B + i*config->inputElementCount + kk * kpartial + k * 512;
                    for (m = 0; m < 512; m++)
                    {
                        acc += weight[m] * input[m];
                    }
                    sum += (int32_t)(acc * config->biasesCompound[i].multiplier);
                    acc = 0;
                }

                input = config->fvBuffers->d0 + j*config->inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->weights1B + i*config->inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; k++)
                {
                    acc += weight[k] * input[k];
                }
                sum += (int32_t)(acc * config->biasesCompound[i].multiplier);

                saturate_store_out(&sum, &config->output[l*config->inputVectorCount + j], config->saturationCount);
                sum = (int64_t)config->output[l*config->inputVectorCount + j]; 
            }
        }
    }
}

void AffineMultiBiasActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al)
{
    uint32_t niters, acc_iters, rem_iters;
    uint32_t i, j, k, l, m;
    int64_t sum;
    int32_t acc;
    uint32_t kk;
    const uint32_t kpartial = hw_buf_size[config->inputVectorCount - 1] / config->inputVectorCount;
    const uint32_t nKpartial = config->inputElementCount / kpartial;

    int16_t const * input;
    int8_t const * weight;

    TransposeConfig transposeConfig = TransposeConfig{ config->inputElementCount, config->inputVectorCount, 
                                                       config->input, config->fvBuffers->d0 };
    TransposeKernelImpl(&transposeConfig);

    for (l = 0; l < al->count; ++l) 
    {
        i = al->indices[l];
        for (j = 0; j < config->inputVectorCount; ++j) 
        {
            sum = config->multiBias[i*config->multiBiasVectorCount];
            for (kk = 0; kk < nKpartial + 1; ++kk) 
            {
                niters = kpartial < config->inputElementCount - kk * kpartial ? kpartial : config->inputElementCount - kk * kpartial;

                acc_iters = niters / 512;
                rem_iters = niters % 512;
                acc = 0;
                for (k = 0; k < acc_iters; ++k)
                {
                    input = config->fvBuffers->d0 + j*config->inputElementCount + kk * kpartial + k * 512;
                    weight = config->weights1B + i*config->inputElementCount + kk * kpartial + k * 512;
                    for (m = 0; m < 512; ++m)
                    {
                        acc += weight[m] * input[m];
                    }
                    sum += acc * config->weightScaleFactors[i].multiplier;
                    acc = 0;
                }

                input = config->fvBuffers->d0 + j*config->inputElementCount + kk * kpartial + acc_iters * 512;
                weight = config->weights1B + i*config->inputElementCount + kk * kpartial + acc_iters * 512;
                for (k = 0; k < rem_iters; ++k)
                {
                    acc += weight[k] * input[k];
                }
                sum += acc * config->weightScaleFactors[i].multiplier;

                saturate_store_out(&sum, &config->output[l*config->inputVectorCount + j], config->saturationCount);
                sum = config->output[l*config->inputVectorCount + j];
            }
        }
    }
}
