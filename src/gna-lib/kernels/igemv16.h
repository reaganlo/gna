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

#define igemm16                 KERNEL(igemm16)
#define igemm16_subset          KERNEL(igemm16_subset)
#define igemm16_mb              KERNEL(igemm16_mb)
#define igemm16_subset_mb       KERNEL(igemm16_subset_mb)
#define igemv16                 KERNEL(igemv16)
#define isbmm16                 KERNEL(isbmm16)
#define transpose16             KERNEL(transpose16)

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
 * @O       output matrix [M,N]
 * @nSat    number of saturations found
 */
void
igemm16(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int16_t*    W,
    const   nn_bias_s*  B,
            int32_t*    O,
            uint32_t*   nSat,
    KernelBuffers*    fvBuffers);

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
 * @O       output matrix [M,N]
 * @AL      active indices list [L]
 * @L       number of active indices
 * @nSat    number of saturations found
 */
void
igemm16_subset(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int16_t*    W,
    const   nn_bias_s*  B,
            int32_t*    O,
    const   uint32_t*   AL,
    const   uint32_t    L,
            uint32_t*   nSat,
    KernelBuffers*    fvBuffers);

/**
 * Calculates affine transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in K rows)
 *  handles multi bias
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @I       input vectors pointer (interleaved) [K,N]
 * @W       weights [M,K]
 * @B       multi bias matrix
 * @BG      multi bias grouping (number of bias vectors)
 * @O       output matrix [M,N]
 * @nSat    number of saturations found
 */
void
igemm16_mb(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int16_t*    W,
    const   nn_bias_s*  B,
    const   uint32_t    BG,
            int32_t*    O,
            uint32_t*   nSat,
    KernelBuffers*    fvBuffers);

/**
 * Calculates affine transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in K rows)
 *  uses active outputs list
 *  handles multi bias
 *
 * @M       number of output elements (out rows)
 * @N       number of input vectors (columns)
 * @K       number of input vector elements (rows)
 * @I       input vectors pointer (interleaved) [K,N]
 * @W       weights [M,K]
 * @B       multi bias matrix
 * @BG      multi bias grouping (number of bias vectors)
 * @O       output matrix [M,N]
 * @AL      active indices list [L]
 * @L       number of active indices
 * @nSat    number of saturations found
 */
void
igemm16_subset_mb(
    const   uint32_t    M,
    const   uint32_t    N,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int16_t*    W,
    const   nn_bias_s*  B,
    const   uint32_t    BG,
    int32_t*    O,
    const   uint32_t*   AL,
    const   uint32_t    L,
            uint32_t*   nSat,
    KernelBuffers*    fvBuffers);

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
igemv16(
const   uint32_t    M,
const   uint32_t    K,
const   int16_t*    I,
const   int16_t*    FB,
const   int16_t*    W,
const   nn_bias_s*  B,
        int32_t*    O,
        uint32_t*   nSat);

/**
 * Calculates diagonal transform on interleaved input vectors
 *  (input vectors in N columns, vector elements in M rows)
 *
 * @M       number of input vector elements (rows)
 * @N       number of input vectors (columns)
 * @I       input vectors pointer (interleaved) [M,N]
 * @W       weights [M]
 * @B       bias [M]
 * @O       output matrix [M,N]
 * @nSat    number of saturations found
 */
void
isbmm16(
    const   uint32_t    M,
    const   uint32_t    N,
    const   int16_t*    W,
    const   int16_t*    I,
    const   nn_bias_s*  B,
            int32_t*    O,
            uint32_t*   nSat);

/**
 * Performs matrix transposition used for interleave/deinterleave input vectors
 * 
 * @M - input rows
 * @N - input columns
 * @I - input vectors
 * @O - output vectors
 */
void transpose16(
    const uint32_t M,
    const uint32_t N,
    const int16_t* I,
          int16_t* O);
 

#ifdef __cplusplus
}
#endif
