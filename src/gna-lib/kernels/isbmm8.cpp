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

#include "igemv8.h"
#include "igemv.h"

void
isbmm8(
    const   uint32_t    M,
    const   uint32_t    N,
    const   int8_t*     W,
    const   int16_t*    I,
    const   nn_bias_c*  B,
            int32_t*    O,
            uint32_t*   nSat)
{
    uint32_t    i, j;
    int64_t     sum;
    int64_t     BiMltp_x_Wi;

    for (i = 0; i < M; i++)
    {
        BiMltp_x_Wi = B[i].multiplier * W[i];
        for (j = 0; j < N; j++) {
            sum =  (int32_t)(B[i].bias + (BiMltp_x_Wi * I[i*N + j]));
            saturate_store_out(&sum, &O[i * N + j], nSat);
        }
    }
}
