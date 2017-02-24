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

void
igemv8(
    const   uint32_t    M,//num_rows N
    const   uint32_t    K,//K1
    const   int16_t*    I,//A2
    const   int16_t*    FB,
    const   int8_t*     W,//X
    const   nn_bias_c*  B,
            int32_t*    O,
            uint32_t*   nSat)
{
    nn_bias_c *bias = const_cast<nn_bias_c*>(B), *bias_end = bias + M;
    int16_t *input = const_cast<int16_t*>(I), *i_end = input + K;
    int16_t *feedback = const_cast<int16_t*>(FB), *fb_end = feedback + M;
    int8_t *weight = const_cast<int8_t*>(W);
    int32_t *out = const_cast<int32_t*>(O);
    int32_t sum;

    for (; bias < bias_end; bias++, out++)
    {
        *out = bias->bias;
        sum = 0;
        input = const_cast<int16_t*>(I);
        feedback = const_cast<int16_t*>(FB);

        for (; input < i_end;)
        {
            sum += *input++ * *weight++;
        }
        for (; feedback < fb_end;)
        {
            sum += *feedback++ * *weight++;
        }

        *out += sum * bias->multiplier;
    }
}