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
    aligned_fv_bufs* fvBuffers)
{
    uint32_t i, j, k, l;
    int64_t sum;
    uint32_t kk;
    uint32_t kpartial;
    uint32_t nKpartial;
    kpartial    = (hw_buf_size[N - 1]) / N;
    nKpartial   = (int32_t)K / kpartial;

    transpose16(K, N, const_cast<int16_t*>(I), fvBuffers->d0);
    int16_t *ptr_in, *ptr_w;

    for (l = 0; l < L; l++) {
        i = AL[l];
        for (j = 0; j < N; j++) {
            sum = B[i];
            for (kk = 0; kk < nKpartial + 1; kk++) {
                ptr_in = fvBuffers->d0 + j*K + kk * kpartial;
                ptr_w = const_cast<int16_t*>(W + i*K + kk * kpartial);
                for (k = 0; (k < kpartial) && (kk*kpartial + k < K); k++) {
                    sum += (int32_t)(ptr_w[k] * ptr_in[k]);
                }
                saturate_store_out(&sum, &O[l*N + j], nSat);
                sum = (int64_t)O[l*N + j]; // load the temp sum
            }
        }
    }
}

void igemm16_subset_mb(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const int16_t *I,
    const int16_t *W,
    const nn_bias_s *B,
    const uint32_t BG,
    int32_t *O,
    const uint32_t *AL,
    const uint32_t L,
    uint32_t *nSat,
    aligned_fv_bufs *fvBuffers)
{
    uint32_t i, j, k, l;
    int64_t sum;
    uint32_t kk;
    const uint32_t kpartial = (hw_buf_size[N - 1]) / N;
    const uint32_t nKpartial = K / kpartial;

    transpose16(K, N, I, fvBuffers->d0);
    const int16_t *ptr_in, *ptr_w;

    for (l = 0; l < L; l++)
    {
        i = AL[l];
        for (j = 0; j < N; j++)
        {
            sum = B[i*BG];
            for (kk = 0; kk < nKpartial + 1; kk++)
            {
                ptr_in = fvBuffers->d0 + j*K + kk * kpartial;
                ptr_w = W + i*K + kk * kpartial;
                for (k = 0; (k < kpartial) && (kk*kpartial + k < K); k++)
                {
                    sum += ptr_w[k] * ptr_in[k];
                }
                saturate_store_out(&sum, &O[l*N + j], nSat);
                sum = O[l*N + j]; // load the temp sum
            }
        }
    }
}
