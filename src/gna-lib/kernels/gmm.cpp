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

#include "gmm.h"
#include "kernel-gmm.h"

#define gmmMaxMix8KernelImpl KERNEL(gmmMaxMix8KernelImpl)
#define gmmMaxMix16KernelImpl KERNEL(gmmMaxMix16KernelImpl)
#define gmmMaxMix8ActiveListKernelImpl KERNEL(gmmMaxMix8ActiveListKernelImpl)
#define gmmMaxMix16ActiveListKernelImpl KERNEL(gmmMaxMix16ActiveListKernelImpl)

void gmmMaxMix8ActiveListKernelImpl(GmmConfig const * const gmmConfig, uint32_t const * const indices)
{
    uint32_t j, k;
    auto gmm = GmmMaxMixConfig{gmmConfig->maximumScore, gmmConfig->inputElementCount, gmmConfig->mixtureComponentCount};

#if OPT_LEVEL == 0 || OPT_LEVEL == 1
    {
        for (j = 0; j < gmmConfig->stateCount; j++)
        {
            k = indices[j];
            gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
            gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
            gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
            gmm.Input = gmmConfig->input;
            gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

            for (uint32_t i = 0; i < gmmConfig->inputVectorCount; i++)
            {
                gmm_maxmix_8u8u_32u(&gmm);
                gmm.Output++;
                gmm.Input += gmm.InputElementOffset;
            }
        }
    }
#elif OPT_LEVEL > 1
    {
        gmm.Input = (uint8_t*)(((unsigned long long)gmmConfig->inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
        uint32_t n = 0;
        uint32_t g = 0;

        // pack feature vectors by 8 features
        // v0[0..7]v1[0..7]vj[0..7]v0[8..15]v1[8..15]...
        if (gmmConfig->inputVectorCount > 1)
        {
            for (n = 0; n < gmm.InputElementCount; n += GMM_FV_COUNT_MAX)
            {
                for (g = 0; g < gmmConfig->inputVectorCount; g++)
                {
                    *((uint64_t*)gmm.Input) = *((uint64_t*)((gmmConfig->input) + g * gmm.InputElementOffset + n));
                    gmm.Input += GMM_FV_COUNT_MAX;
                }
            }
            gmm.Input = (uint8_t*)(((unsigned long long)gmmConfig->inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
        }
        else
        {
            gmm.Input = gmmConfig->input;
        }

        switch (gmmConfig->inputVectorCount)
        {
        case 1:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g1(&gmm);
            }
            break;
        case 2:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g2(&gmm);
            }
            break;
        case 3:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g3(&gmm);
            }
            break;
        case 4:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = (uint32_t*)gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g4(&gmm);
            }
            break;
        case 5:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g5(&gmm);
            }
            break;
        case 6:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g6(&gmm);
            }
            break;
        case 7:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g7(&gmm);
            }
            break;
        case 8:

            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                k = indices[j];
                gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + k * gmmConfig->varSetOffsetSize;
                gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g8(&gmm);
            }
            break;
        }
    }
#endif
}

void gmmMaxMix16ActiveListKernelImpl(GmmConfig const * const gmmConfig, uint32_t const * const indices)
{
    uint32_t i, j, k;
    auto gmm = GmmMaxMixConfig{gmmConfig->maximumScore, gmmConfig->inputElementCount, gmmConfig->mixtureComponentCount};

    gmm.Output = gmmConfig->output;

    for (j = 0; j < gmmConfig->stateCount; j++)
    {
        k = indices[j];
        gmm.Means = gmmConfig->data->meanValues + k * gmmConfig->meanSetOffsetSize;
        gmm.Vars16 = gmmConfig->data->inverseCovariancesForMaxMix16 + k * gmmConfig->varSetOffsetSize / GMM_COVARIANCE_SIZE_MAX;
        gmm.Gconst = (uint32_t*)((uint8_t*)gmmConfig->data->gaussianConstants + k * gmmConfig->gaussConstSetOffsetSize);
        gmm.Input = gmmConfig->input;

        for (i = 0; i < gmmConfig->inputVectorCount; i++)
        {
            gmm_maxmix_8u16u_32u(&gmm);
            gmm.Output++;
            gmm.Input += gmm.InputElementOffset;
        }
    }
}

void gmmMaxMix8KernelImpl(GmmConfig const * const gmmConfig)
{
    uint32_t j;
    uint32_t const meanOffset = gmmConfig->meanSetOffsetSize / GMM_MEAN_VALUE_SIZE;;
    uint32_t const varOffset = gmmConfig->varSetOffsetSize / (GNA_MAXMIX8 + 1);;
    uint32_t const gConstOffset = gmmConfig->gaussConstSetOffsetSize / GMM_CONSTANTS_SIZE;
    auto gmm = GmmMaxMixConfig{gmmConfig->maximumScore, gmmConfig->inputElementCount, gmmConfig->mixtureComponentCount};

#if OPT_LEVEL == 0 || OPT_LEVEL == 1
    {
        for (j = 0; j < gmmConfig->stateCount; j++)
        {
            gmm.Input = gmmConfig->input;
            gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
            gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
            gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
            gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

            for (uint32_t i = 0; i < gmmConfig->inputVectorCount; i++)
            {
                gmm_maxmix_8u8u_32u(&gmm);
                gmm.Output++;
                gmm.Input += gmm.InputElementOffset;
            }
        }
    }
#elif OPT_LEVEL > 1
    {
        gmm.Input = (uint8_t*)(((unsigned long long)gmmConfig->inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull); // aligned to GMM_FV_MEM_ALIGN bytes
        uint32_t n;
        uint32_t g;

        // pack feature vectors by 8 features
        // v0[0..7]v1[0..7]vj[0..7]v0[8..15]v1[8..15]...
        if (gmmConfig->inputVectorCount > 1)
        {
            for (n = 0; n < gmm.InputElementCount; n += GMM_FV_COUNT_MAX)
            {
                for (g = 0; g < gmmConfig->inputVectorCount; g++)
                {
                    *((uint64_t*)gmm.Input) = *((uint64_t*)((gmmConfig->input) + g * gmm.InputElementOffset + n));
                    gmm.Input += GMM_FV_COUNT_MAX;
                }
            }
            gmm.Input = (uint8_t*)(((unsigned long long)gmmConfig->inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
        }
        else
        {
            gmm.Input = gmmConfig->input;
        }
        switch (gmmConfig->inputVectorCount)
        {
        case 1:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g1(&gmm);

            }
            break;
        case 2:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g2(&gmm);
            }
            break;
        case 3:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g3(&gmm);
            }
            break;
        case 4:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g4(&gmm);
            }
            break;
        case 5:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g5(&gmm);
            }
            break;
        case 6:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g6(&gmm);
            }
            break;
        case 7:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g7(&gmm);
            }
            break;
        case 8:
            for (j = 0; j < gmmConfig->stateCount; j++)
            {
                gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
                gmm.Vars = gmmConfig->data->inverseCovariancesForMaxMix8 + j*varOffset;
                gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
                gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

                gmm_maxmix_8u8u_32u_g8(&gmm);
            }
            break;
        }
    }
#endif
}

void gmmMaxMix16KernelImpl(GmmConfig const * const gmmConfig)
{
    uint32_t i, j;
    uint32_t const meanOffset = gmmConfig->meanSetOffsetSize / GMM_MEAN_VALUE_SIZE;
    uint32_t const varOffset = gmmConfig->varSetOffsetSize / (GNA_MAXMIX16 + 1);
    uint32_t const gConstOffset = gmmConfig->gaussConstSetOffsetSize / GMM_CONSTANTS_SIZE;
    auto gmm = GmmMaxMixConfig{gmmConfig->maximumScore, gmmConfig->inputElementCount, gmmConfig->mixtureComponentCount};

    for (j = 0; j < gmmConfig->stateCount; j++)
    {
        gmm.Input = gmmConfig->input;
        gmm.Means = gmmConfig->data->meanValues + j*meanOffset;
        gmm.Vars16 = gmmConfig->data->inverseCovariancesForMaxMix16 + j*varOffset;
        gmm.Gconst = gmmConfig->data->gaussianConstants + j*gConstOffset;
        gmm.Output = gmmConfig->output + j*gmmConfig->inputVectorCount;

        for (i = 0; i < gmmConfig->inputVectorCount; i++)
        {
            gmm_maxmix_8u16u_32u(&gmm);
            gmm.Output++;
            gmm.Input += gmm.InputElementOffset;
        }
    }
}

GmmKernel KERNEL(gmmKernel)
{
    gmmMaxMix8KernelImpl,
    gmmMaxMix16KernelImpl,
    gmmMaxMix8ActiveListKernelImpl,
    gmmMaxMix16ActiveListKernelImpl
};
