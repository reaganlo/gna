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

#include "KernelArguments.h"
#include "KernelMacros.h"

#include "gna-api-types-gmm.h"

#define gmmMaxMix8KernelImpl KERNEL(gmmMaxMix8KernelImpl)
#define gmmMaxMix16KernelImpl KERNEL(gmmMaxMix16KernelImpl)
#define gmmMaxMix8ActiveListKernelImpl KERNEL(gmmMaxMix8ActiveListKernelImpl)
#define gmmMaxMix16ActiveListKernelImpl KERNEL(gmmMaxMix16ActiveListKernelImpl)
#define checkScoresSaturation KERNEL(checkScoresSaturation)
#define calculateOffsets KERNEL(calculateOffsets)

inline void checkScoresSaturation(const uint32_t& nGMMs, const uint32_t& nVectors, const uint32_t * pS,
    const uint32_t& maximumScore, uint32_t& nSaturated)
{
    for (auto i = uint32_t{ 0 }; i < nGMMs * nVectors; i++)
    {
        if (maximumScore == *pS)
        {
            nSaturated++;
            return;
        }
        pS++;
    }
}

inline void calculateOffsets(GmmConfig * const & gmmConfig, uint32_t * const & output,
    uint32_t & j, uint32_t & k, GmmConfig & gmm)
{
    gmm.Means = gmmConfig->Means + k * gmmConfig->MeanSetOffsetSize;
    gmm.Vars = gmmConfig->Vars + k * gmmConfig->VarSetOffsetSize;
    gmm.Gconst = gmmConfig->Gconst + k * gmmConfig->GaussConstSetOffsetSize / GMM_CONSTANTS_SIZE;
    gmm.Output = output + j * gmmConfig->InputVectorCount;
}

void gmmMaxMix8ActiveListKernelImpl(ExecutionKernelConfig<GmmConfig> const * const config, AffineConfigAl al)
{
    auto const gmmConfig = &config->RequestConfig->Transform;
    auto const * const input = reinterpret_cast<uint8_t *>(
        config->RequestConfig->Buffers[GNA::InputOperandIndex]);
    auto * const output = reinterpret_cast<uint32_t *>(
        config->RequestConfig->Buffers[GNA::OutputOperandIndex]);
    auto const indices = al.indices;
    auto const StateCount = al.count;
    uint32_t j, k;
    auto gmm = *gmmConfig;

#if OPT_LEVEL == 0 || OPT_LEVEL == 1
    {
        for (j = 0; j < StateCount; j++)
        {
            k = indices[j];
            gmm.Input = input;
            calculateOffsets(gmmConfig, output, j, k, gmm);

            for (uint32_t i = 0; i < gmmConfig->InputVectorCount; i++)
            {
                gmm_maxmix_8u8u_32u(&gmm);
                gmm.Output++;
                gmm.Input += gmm.InputElementOffset;
            }
        }
    }
#elif OPT_LEVEL > 1
    {
        auto * const inputScratchPad = reinterpret_cast<uint8_t *>(config->Intermediate->d0);
        gmm.Input = (uint8_t*)(((unsigned long long)inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
        uint32_t n = 0;
        uint32_t g = 0;

        // pack feature vectors by 8 features
        // v0[0..7]v1[0..7]vj[0..7]v0[8..15]v1[8..15]...
        if (gmmConfig->InputVectorCount > 1)
        {
            for (n = 0; n < gmm.InputElementCount; n += GMM_FV_COUNT_MAX)
            {
                for (g = 0; g < gmmConfig->InputVectorCount; g++)
                {
                    *((uint64_t*)gmm.Input) = *((uint64_t*)((input)+g * gmm.InputElementOffset + n));
                    gmm.Input += GMM_FV_COUNT_MAX;
                }
            }
            gmm.Input = (uint8_t*)(((unsigned long long)inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
        }
        else
        {
            gmm.Input = input;
        }

        switch (gmmConfig->InputVectorCount)
        {
        case 1:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g1(&gmm);
            }
            break;
        case 2:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g2(&gmm);
            }
            break;
        case 3:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g3(&gmm);
            }
            break;
        case 4:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g4(&gmm);
            }
            break;
        case 5:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g5(&gmm);
            }
            break;
        case 6:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g6(&gmm);
            }
            break;
        case 7:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g7(&gmm);
            }
            break;
        case 8:

            for (j = 0; j < StateCount; j++)
            {
                k = indices[j];
                calculateOffsets(gmmConfig, output, j, k, gmm);

                gmm_maxmix_8u8u_32u_g8(&gmm);
            }
            break;
        }
    }
#endif

    checkScoresSaturation(StateCount, gmmConfig->InputVectorCount,
        output, gmmConfig->MaxScore, *config->SaturationCount);
}

void gmmMaxMix16ActiveListKernelImpl(ExecutionKernelConfig<GmmConfig> const * const config, AffineConfigAl al)
{
    auto const gmmConfig = &config->RequestConfig->Transform;
    auto const * const input = reinterpret_cast<uint8_t *>(
        config->RequestConfig->Buffers[GNA::InputOperandIndex]);
    auto * const output = reinterpret_cast<uint32_t *>(
        config->RequestConfig->Buffers[GNA::OutputOperandIndex]);
    auto const indices = al.indices;
    auto const StateCount = al.count;
    uint32_t i, j, k;
    auto gmm = *gmmConfig;

    gmm.Output = output;

    for (j = 0; j < StateCount; j++)
    {
        k = indices[j];
        gmm.Input = input;
        calculateOffsets(gmmConfig, output, j, k, gmm);

        for (i = 0; i < gmmConfig->InputVectorCount; i++)
        {
            gmm_maxmix_8u16u_32u(&gmm);
            gmm.Output++;
            gmm.Input += gmm.InputElementOffset;
        }
    }

    checkScoresSaturation(StateCount, gmmConfig->InputVectorCount,
        output, gmmConfig->MaxScore, *config->SaturationCount);
}

void gmmMaxMix8KernelImpl(ExecutionKernelConfig<GmmConfig> const * const config)
{
    auto const gmmConfig = &config->RequestConfig->Transform;
    auto const * const input = reinterpret_cast<uint8_t *>(
        config->RequestConfig->Buffers[GNA::InputOperandIndex]);
    auto * const output = reinterpret_cast<uint32_t *>(
        config->RequestConfig->Buffers[GNA::OutputOperandIndex]);
    uint32_t j;
    auto gmm = *gmmConfig;

#if OPT_LEVEL == 0 || OPT_LEVEL == 1
    {
        for (j = 0; j < gmmConfig->StateCount; j++)
        {
            gmm.Input = input;
            calculateOffsets(gmmConfig, output, j, j, gmm);

            for (uint32_t i = 0; i < gmmConfig->InputVectorCount; i++)
            {
                gmm_maxmix_8u8u_32u(&gmm);
                gmm.Output++;
                gmm.Input += gmm.InputElementOffset;
            }
        }
    }
#elif OPT_LEVEL > 1
    {
        auto * const inputScratchPad = reinterpret_cast<uint8_t *>(config->Intermediate->d0);
        gmm.Input = (uint8_t*)(((unsigned long long)inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull); // aligned to GMM_FV_MEM_ALIGN bytes
        uint32_t n;
        uint32_t g;

        // pack feature vectors by 8 features
        // v0[0..7]v1[0..7]vj[0..7]v0[8..15]v1[8..15]...
        if (gmmConfig->InputVectorCount > 1)
        {
            for (n = 0; n < gmm.InputElementCount; n += GMM_FV_COUNT_MAX)
            {
                for (g = 0; g < gmmConfig->InputVectorCount; g++)
                {
                    *((uint64_t*)gmm.Input) = *((uint64_t*)((input)+g * gmm.InputElementOffset + n));
                    gmm.Input += GMM_FV_COUNT_MAX;
                }
            }
            gmm.Input = (uint8_t*)(((unsigned long long)inputScratchPad + GMM_FV_MEM_ALIGN) & 0xffffffffffffffc0ull);
        }
        else
        {
            gmm.Input = input;
        }
        switch (gmmConfig->InputVectorCount)
        {
        case 1:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g1(&gmm);

            }
            break;
        case 2:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g2(&gmm);
            }
            break;
        case 3:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g3(&gmm);
            }
            break;
        case 4:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g4(&gmm);
            }
            break;
        case 5:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g5(&gmm);
            }
            break;
        case 6:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g6(&gmm);
            }
            break;
        case 7:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g7(&gmm);
            }
            break;
        case 8:
            for (j = 0; j < gmmConfig->StateCount; j++)
            {
                calculateOffsets(gmmConfig, output, j, j, gmm);

                gmm_maxmix_8u8u_32u_g8(&gmm);
            }
            break;
        }
    }
#endif

    checkScoresSaturation(gmmConfig->StateCount, gmmConfig->InputVectorCount,
        output, gmmConfig->MaxScore, *config->SaturationCount);
}

void gmmMaxMix16KernelImpl(ExecutionKernelConfig<GmmConfig> const * const config)
{
    auto const gmmConfig = &config->RequestConfig->Transform;
    auto const * const input = reinterpret_cast<uint8_t *>(
        config->RequestConfig->Buffers[GNA::InputOperandIndex]);
    auto * const output = reinterpret_cast<uint32_t *>(
        config->RequestConfig->Buffers[GNA::OutputOperandIndex]);
    uint32_t i, j;
    auto gmm = *gmmConfig;

    for (j = 0; j < gmmConfig->StateCount; j++)
    {
        gmm.Input = input;
        calculateOffsets(gmmConfig, output, j, j, gmm);

        for (i = 0; i < gmmConfig->InputVectorCount; i++)
        {
            gmm_maxmix_8u16u_32u(&gmm);
            gmm.Output++;
            gmm.Input += gmm.InputElementOffset;
        }
    }

    checkScoresSaturation(gmmConfig->StateCount, gmmConfig->InputVectorCount,
        output, gmmConfig->MaxScore, *config->SaturationCount);
}

GmmKernel KERNEL(gmmKernel)
{
    gmmMaxMix8KernelImpl,
    gmmMaxMix16KernelImpl,
    gmmMaxMix8ActiveListKernelImpl,
    gmmMaxMix16ActiveListKernelImpl
};
