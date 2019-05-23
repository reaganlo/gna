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

#include "KernelArguments.h"

#include "common.h"
#include "gna-api-types-xnn.h"

#include <cstdint>

void RecurrentKernelImpl1B(RecurrentConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd = bias + (config->outputElementCount);
    int16_t const * input;
    int16_t * feedback;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;
    uint32_t kparts = config->inputElementCount / config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_rem = config->inputElementCount % config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t middle_fill = config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX] - kpart_rem;
    uint32_t middle_part = (config->outputElementCount < middle_fill) ? config->outputElementCount : middle_fill;
    uint32_t mm = config->outputElementCount - middle_part;
    uint32_t mparts = mm / config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t mpart_rem = mm % config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];

    for (; bias < biasEnd; bias++)
    {
        sum = bias->bias;
        input = config->input;
        feedback = config->feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                sum += *input++ * *weight++ * bias->multiplier;
            }
            saturate_store_out(&sum, output, config->execution->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++ * bias->multiplier;
        }

        for (i = 0; i < middle_part; i++)
        {
            sum += *feedback++ * *weight++ * bias->multiplier;
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                sum += *feedback++ * *weight++ * bias->multiplier;
            }

            saturate_store_out(&sum, output, config->execution->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            sum += *feedback++ * *weight++ * bias->multiplier;
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);
        output++;
    }
}

void RecurrentKernelImpl1B2B(RecurrentConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;
    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd= bias + (config->outputElementCount);
    int16_t const * input;
    int8_t * feedback;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;
    uint32_t kparts = config->inputElementCount / config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t kpart_rem = config->inputElementCount % config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t middle_fill = config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX] - kpart_rem;
    uint32_t middle_part = (config->outputElementCount < middle_fill) ? config->outputElementCount : middle_fill;
    uint32_t mm = config->outputElementCount - middle_part;
    uint32_t mparts = mm / config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];
    uint32_t mpart_rem = mm % config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX];

    for (; bias < biasEnd; bias++)
    {
        sum = bias->bias;
        input = config->input;
        feedback = (int8_t *)config->feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                sum += *input++ * *weight++ * bias->multiplier;
            }
            saturate_store_out(&sum, output, config->execution->SaturationCount);
            sum = *output;
        }

        for(i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++ * bias->multiplier;
        }

        for (i = 0; i < middle_part; i++)
        {
            if (config->bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++ * bias->multiplier;
            }
            else if (config->bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++ * bias->multiplier;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->execution->BufferElementCount[0 + XNN_N_GROUP_MAX]; j++)
            {
                if (config->bytesPerOutput == 1)
                {
                    sum += *feedback++ * *weight++ * bias->multiplier;
                }
                else if (config->bytesPerOutput == 2)
                {
                    sum += *(int16_t*)feedback * *weight++ * bias->multiplier;
                    feedback += 2;
                }
            }

            saturate_store_out(&sum, output, config->execution->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            if (config->bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++ * bias->multiplier;
            }
            else if (config->bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++ * bias->multiplier;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);
        output++;
    }
}

void RecurrentKernelImpl1B1B(RecurrentConfig const * const config)
{
    uint32_t i;
    uint32_t j;
    int64_t sum = 0;

    int8_t const * bias = (int8_t*)config->biasesSimple;
    int8_t const * const biasEnd = bias + (config->outputElementCount * config->bytesPerBias);
    int8_t const * input;
    int8_t * feedback;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;
    uint32_t kparts = config->inputElementCount / config->execution->BufferElementCount[0];
    uint32_t kpart_rem = config->inputElementCount % config->execution->BufferElementCount[0];
    uint32_t middle_fill = config->execution->BufferElementCount[0] - kpart_rem;
    uint32_t middle_part = (config->outputElementCount < middle_fill) ? config->outputElementCount : middle_fill;
    uint32_t mm = config->outputElementCount - middle_part;
    uint32_t mparts = mm / config->execution->BufferElementCount[0];
    uint32_t mpart_rem = mm % config->execution->BufferElementCount[0];

    for (; bias < biasEnd; bias += config->bytesPerBias)
    {
        if (config->bytesPerBias == 1)
        {
            sum = *bias;
        }
        else if (config->bytesPerBias == 2)
        {
            sum = *(int16_t*)bias;
        }
        else if (config->bytesPerBias == 4)
        {
            sum = *(int32_t*)bias;
        }

        input = (int8_t*)config->input;
        feedback = (int8_t*)config->feedbackBuffer;

        for (i = 0; i < kparts; i++)
        {
            for (j = 0; j < config->execution->BufferElementCount[0]; j++)
            {
                sum += *input++ * *weight++;
            }
            saturate_store_out(&sum, output, config->execution->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < kpart_rem; i++)
        {
            sum += *input++ * *weight++;
        }

        for (i = 0; i < middle_part; i++)
        {
            if (config->bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++;
            }
            else if (config->bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);
        sum = *output;

        for (i = 0; i < mparts; i++)
        {
            for (j = 0; j < config->execution->BufferElementCount[0]; j++)
            {
                if (config->bytesPerOutput == 1)
                {
                    sum += *feedback++ * *weight++;
                }
                else if (config->bytesPerOutput == 2)
                {
                    sum += *(int16_t*)feedback * *weight++;
                    feedback += 2;
                }
            }

            saturate_store_out(&sum, output, config->execution->SaturationCount);
            sum = *output;
        }

        for (i = 0; i < mpart_rem; i++)
        {
            if (config->bytesPerOutput == 1)
            {
                sum += *feedback++ * *weight++;
            }
            else if (config->bytesPerOutput == 2)
            {
                sum += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }

        saturate_store_out(&sum, output, config->execution->SaturationCount);
        output++;
    }
}
