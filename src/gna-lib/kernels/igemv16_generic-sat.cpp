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

void RecurrentKernelImpl2B(RecurrentConfig const * const config)
{
    uint32_t i,j;
    int64_t sum;

    nn_bias_s const * bias = config->biasesSimple; 
    nn_bias_s const * const biasEnd= bias + config->outputElementCount;
    int16_t const * input;
    int16_t * feedback;
    int16_t const * weight = config->weights2B;
    int32_t * output = config->output;
    uint32_t kparts = config->inputElementCount / hw_buf_size[0];
    uint32_t kpart_rem = config->inputElementCount % hw_buf_size[0];
    uint32_t middle_fill = hw_buf_size[0] - kpart_rem;
    uint32_t middle_part = (config->outputElementCount < middle_fill) ? config->outputElementCount : middle_fill;
    uint32_t mm = config->outputElementCount - middle_part;
    uint32_t mparts = mm / hw_buf_size[0];
    uint32_t mpart_rem = mm % hw_buf_size[0];

    for (; bias < biasEnd; bias++)
    {
        sum = *bias;
        input = config->input;
        feedback = config->feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < hw_buf_size[0]; j++)
            {
                sum += *input++ * *weight++;
            }
            saturate_store_out(&sum, output, config->saturationCount);
            sum = *output;
        }

        for(i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++;
        }

        for (i = 0; i < middle_part; i++)
        {
            sum += *feedback++ * *weight++;
        }

        saturate_store_out(&sum, output, config->saturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < hw_buf_size[0]; j++)
            {
                sum += *feedback++ * *weight++;
            }

            saturate_store_out(&sum, output, config->saturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            sum += *feedback++ * *weight++;
        }

        saturate_store_out(&sum, output, config->saturationCount);
        output++;
    }
}