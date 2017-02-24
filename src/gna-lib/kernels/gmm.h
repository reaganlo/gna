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

#include <stdint.h>

// Definition of gmm_maxmix_8u8u_32u functions arguments list
#define _GMM8_ARGS \
    const uint8_t  *pFeat,          \
    const uint8_t  *pMeans,         \
    const uint8_t  *pVars,          \
    const uint32_t *pGconst,        \
          uint32_t  ScoreLimit32u,  \
          uint32_t  nVecElements,   \
          uint32_t  nMixtures
// Definition of gmm_maxmix_8u8u_32u_g1? functions arguments list
#define _GMM8_MAXMIX_ARGS \
    const uint8_t  *pFeat,          \
    const uint8_t  *pMeans,         \
    const uint8_t  *pVars,          \
    const uint32_t *pGconst,        \
          uint32_t  ScoreLimit32u,  \
          uint32_t  nVecElements,   \
          uint32_t  nMixtures,      \
          uint32_t* pScores 
// Definition of gmm_maxmix_8u16u_32u functions arguments list
#define _GMM16_ARGS \
    const uint8_t  *pFeat,          \
    const uint8_t  *pMeans,         \
    const uint16_t *pVars,          \
    const uint32_t *pGconst,        \
          uint32_t  ScoreLimit32u,  \
          uint32_t  nVecElements,   \
          uint32_t  nMixtures

/**
 * gmm_maxmix_8u8u_32u function pointer type
 */
typedef
uint32_t
(*Gmm8Fn)(
    _GMM8_ARGS);

/**
 * gmm_maxmix_8u16u_32u function pointer type
 */
typedef
uint32_t
(*Gmm16Fn)(
    _GMM16_ARGS);

/**
 * gmm_maxmix_8u8u_32u_g1? function pointer type
 */
typedef 
void
(*Gmm8MxFn)(
    _GMM8_MAXMIX_ARGS);

/**
 * GMM kernel provider
 *
 *  Contains GMM kernel function pointers for selected acceleration
 */
typedef struct _GmmKernel
{
    Gmm8Fn      GMM8;
    Gmm16Fn     GMM16;
    Gmm8MxFn    GMM8_MAXMIX_G1;
    Gmm8MxFn    GMM8_MAXMIX_G2;
    Gmm8MxFn    GMM8_MAXMIX_G3;
    Gmm8MxFn    GMM8_MAXMIX_G4;
    Gmm8MxFn    GMM8_MAXMIX_G5;
    Gmm8MxFn    GMM8_MAXMIX_G6;
    Gmm8MxFn    GMM8_MAXMIX_G7;
    Gmm8MxFn    GMM8_MAXMIX_G8;

} GmmKernel;                        // GMM kernel provider

/**
 * Export list of available GMM kernels providers
 */

/** generic GMM kernel provider */
extern GmmKernel gmmKernel_generic;

/** sse4.2 accelerated GMM kernel provider */
extern GmmKernel gmmKernel_sse4;

/** avx1 accelerated GMM kernel provider */
extern GmmKernel gmmKernel_avx1;

/** avx2 accelerated GMM kernel provider */
extern GmmKernel gmmKernel_avx2;
