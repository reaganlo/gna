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
igemv16(
    const   uint32_t    M,
    const   uint32_t    K,
    const   int16_t*    I,
    const   int16_t*    FB,
    const   int16_t*    W,
    const   nn_bias_s*  B,
            int32_t*    O,
            uint32_t*   nSat)
{
    uint32_t i,j;
    int64_t sum;

    nn_bias_s *bias = const_cast<nn_bias_s*>(B), *bias_end = bias + M;
    int16_t *input;
    int16_t *feedback;
    int16_t *weight = const_cast<int16_t*>(W);
    int32_t *out = const_cast<int32_t*>(O);
    uint32_t kparts = K / hw_buf_size[0];
    uint32_t kpart_rem = K % hw_buf_size[0];
    uint32_t middle_fill = hw_buf_size[0] - kpart_rem;
    uint32_t middle_part = (M < middle_fill) ? M : middle_fill;
    uint32_t mm = M - middle_part;
    uint32_t mparts = mm / hw_buf_size[0];
    uint32_t mpart_rem = mm % hw_buf_size[0];

    for (; bias < bias_end; bias++)
    {
        sum = *bias;
        input = const_cast<int16_t*>(I);
        feedback = const_cast<int16_t*>(FB);

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < hw_buf_size[0]; j++)
            {
                sum += *input++ * *weight++;
            }
            saturate_store_out(&sum, out, nSat);
            sum = *out;
        }

        for(i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++;
        }

        for (i = 0; i < middle_part; i++)
        {
            sum += *feedback++ * *weight++;
        }

        saturate_store_out(&sum, out, nSat);
        sum = *out;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < hw_buf_size[0]; j++)
            {
                sum += *feedback++ * *weight++;
            }

            saturate_store_out(&sum, out, nSat);
            sum = *out;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            sum += *feedback++ * *weight++;
        }

        saturate_store_out(&sum, out, nSat);
        out++;
    }
}