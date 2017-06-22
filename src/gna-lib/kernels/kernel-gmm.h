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

#pragma once

#include "common.h"
#include "KernelMacros.h"

#define gmm_maxmix_8u8u_32u     KERNEL(gmm_maxmix_8u8u_32u)
#define gmm_maxmix_8u16u_32u    KERNEL(gmm_maxmix_8u16u_32u)
#if OPT_LEVEL > 1
#define gmm_maxmix_8u8u_32u_g1  KERNEL(gmm_maxmix_8u8u_32u_g1)
#define gmm_maxmix_8u8u_32u_g2  KERNEL(gmm_maxmix_8u8u_32u_g2)
#define gmm_maxmix_8u8u_32u_g3  KERNEL(gmm_maxmix_8u8u_32u_g3)
#define gmm_maxmix_8u8u_32u_g4  KERNEL(gmm_maxmix_8u8u_32u_g4)
#define gmm_maxmix_8u8u_32u_g5  KERNEL(gmm_maxmix_8u8u_32u_g5)
#define gmm_maxmix_8u8u_32u_g6  KERNEL(gmm_maxmix_8u8u_32u_g6)
#define gmm_maxmix_8u8u_32u_g7  KERNEL(gmm_maxmix_8u8u_32u_g7)
#define gmm_maxmix_8u8u_32u_g8  KERNEL(gmm_maxmix_8u8u_32u_g8)
#endif

// GMM kernel implementation arguments
struct GmmMaxMixConfig
{
    GmmMaxMixConfig(uint32_t const scoreLimit, uint32_t const inputElementCount, uint32_t const mixtureCount) :
        MinScore{scoreLimit},
        InputElementCount{inputElementCount},
        InputElementOffset{ALIGN64(InputElementCount)},
        MixtureCount{mixtureCount},
        Means{nullptr},
        Vars{nullptr},
        Gconst{nullptr},
        Input{nullptr},
        Output{nullptr}
    {}

    uint32_t const MinScore;
    uint32_t const InputElementCount;
    uint32_t const InputElementOffset;
    uint32_t const MixtureCount;
    uint8_t const * Means;
    union
    {
    uint8_t const * Vars;
    uint16_t const * Vars16;
    };
    uint32_t const * Gconst;
    uint8_t const * Input;
    uint32_t * Output;
};


void gmm_maxmix_8u8u_32u(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u16u_32u(GmmMaxMixConfig const * const config);

#if OPT_LEVEL > 1
void gmm_maxmix_8u8u_32u_g1(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g2(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g3(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g4(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g5(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g6(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g7(GmmMaxMixConfig const * const config);

void gmm_maxmix_8u8u_32u_g8(GmmMaxMixConfig const * const config);
#endif
