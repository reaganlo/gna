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

#include "KernelMacros.h"
#include "common.h"

#define igemm8                 KERNEL(igemm8)
#define igemm8_subset          KERNEL(igemm8_subset)
#define igemm8_mb              KERNEL(igemm8_mb)
#define igemm8_subset_mb       KERNEL(igemm8_subset_mb)
#define igemv8                 KERNEL(igemv8)
#define isbmm8                 KERNEL(isbmm8)

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif

/**
 * Calculates affine transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in K rows)
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @I       input vectors pointer (interleaved) [K,N]
 * @W       weights [M,K]
 * @B       bias vector [M]
 * @O       output matrix [M,N]
 * @nSat    number of saturations found
 */
void
igemm8(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int8_t*     W,
    const   nn_bias_c*  B,
            int32_t*    O,
            uint32_t*   nSat,
    aligned_fv_bufs*    fvBuffers);

/**
 * Calculates affine transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in K rows)
 *  uses active outputs list
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @I       input vectors pointer (interleaved) [K,N]
 * @W       weights [M,K]
 * @B       bias vector [M]
 * @O       output matrix [M,N]
 * @AL      active indices list [L]
 * @L       number of active indices
 * @nSat    number of saturations found
 */
void
igemm8_subset(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int8_t*     W,
    const   nn_bias_c*  B,
            int32_t*    O,
    const   uint32_t*   AL,
    const   uint32_t    L,
            uint32_t*   nSat,
    aligned_fv_bufs*    fvBuffers);

/**
 * Calculates affine transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in K rows)
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @I       input vectors pointer (interleaved) [K,N]
 * @W       weights [M,K]
 * @B       multi bias matrix
 * @CB      compound bias vector [M]
 * @BG      multi bias grouping (number of bias vectors)
 * @O       output matrix [M,N]
 * @nSat    number of saturations found
 */
void
igemm8_mb(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int8_t*     W,
    const   nn_bias_s*  B,
    const   uint32_t    BG,
    const   nn_bias_c*  CB,
            int32_t*    O,
            uint32_t*   nSat,
    aligned_fv_bufs*    fvBuffers);

/**
 * Calculates affine transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in K rows)
 *  uses active outputs list
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @I       input vectors pointer (interleaved) [K,N]
 * @W       weights [M,K]
 * @B       multi bias matrix
 * @CB      compound bias vector [M]
 * @BG      multi bias grouping (number of bias vectors)
 * @O       output matrix [M,N]
 * @AL      active indices list [L]
 * @L       number of active indices
 * @nSat    number of saturations found
 */
void
igemm8_subset_mb(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int8_t*     W,
    const   nn_bias_s*  B,
    const   uint32_t    BG,
    const   nn_bias_c*  CB,
            int32_t*    O,
    const   uint32_t*   AL,
    const   uint32_t    L,
            uint32_t*   nSat,
    aligned_fv_bufs*    fvBuffers);

/**
 * Calculates recurrent transform on flat input vectors
 *  (input vectors in N rows, vector elements in K columns)
 *
 * @M       number of output elements (out cols)
 * @K       number of input vector elements (cols)
 * @I       input vectors pointer (flat) [N,K]
 * @FB      feedback vectors pointer (flat) [N,M]
 * @W       weights [M,K+M]
 * @B       bias vector [M]
 * @O       output matrix [N,M]
 * @nSat    number of saturations found
 */
void
igemv8(
    const   uint32_t    M,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int16_t*    FB,
    const   int8_t*     W,
    const   nn_bias_c*  B,
            int32_t*    O,
            uint32_t*   nSat);

/**
 * Calculates affine transform on interleaved input vectors
 * with diagonal weight matrix W[M]
 *  (input vectors in N columns, vector elements in K rows)
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @W       weights [M]
 * @I       input vectors pointer (interleaved) [K,N]
 * @B       bias vector [M]
 * @O       output matrix [M,N]
 * @nSat    number of saturations found
 */
void
isbmm8(
    const   uint32_t    M,
    const   uint32_t    N,
    const   int8_t*     W,
    const   int16_t*    I,
    const   nn_bias_c*  B,
            int32_t*    O,
            uint32_t*   nSat);

#ifdef __cplusplus
}
#endif
