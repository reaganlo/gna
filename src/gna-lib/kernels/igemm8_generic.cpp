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
    aligned_fv_bufs* fvBuffers)
{
	uint32_t i,j,k;
    int8_t *ptr_w = const_cast<int8_t*>(W);
    int16_t *ptr_in;
    int32_t *ptr_out = O;
    nn_bias_c *ptr_b = const_cast<nn_bias_c*>(B),
              *end_b = ptr_b + M;

    transpose16(K, N, const_cast<int16_t*>(I), fvBuffers->d0);

    for (; ptr_b < end_b;)
    {
        ptr_in = fvBuffers->d0;
        for (j = 0; j < N; j++)
        {
            *ptr_out = 0;
            for (k = 0; k < K; k++)
            {
                *ptr_out += ptr_w[k] * *ptr_in++;
            }
            *ptr_out *= ptr_b->multiplier;
            *ptr_out++ += ptr_b->bias;
        }
        ptr_w += K;
        ptr_b++;
    }
}

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
    aligned_fv_bufs*    fvBuffers)
{
    uint32_t j, k;
    int16_t *ptr_in;
    const nn_bias_c * const end_b = CB + M;

    transpose16(K, N, I, fvBuffers->d0);

    for (; CB < end_b;)
    {
        ptr_in = fvBuffers->d0;
        for (j = 0; j < N; ++j)
        {
            *O = 0;
            for (k = 0; k < K; ++k)
            {
                *O += W[k] * *ptr_in++;
            }
            *O *= CB->multiplier;
            *O++ += *B;
        }
        W += K;
        B += BG;
        CB++;
    }
}
