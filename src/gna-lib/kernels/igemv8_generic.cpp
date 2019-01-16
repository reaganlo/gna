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

void RecurrentKernelImpl1B(RecurrentConfig const * const config)
{
    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;
    int16_t const * input = config->input;
    int16_t const * const inputEnd = input + config->inputElementCount;
    int16_t * feedback = config->feedbackBuffer;
    int16_t const * const feedbackEnd = feedback + config->outputElementCount;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;
    int32_t sum;

    for (; bias < biasEnd; bias++, output++)
    {
        *output = bias->bias;
        sum = 0;
        input = config->input;
        feedback = config->feedbackBuffer;

        for (; input < inputEnd;)
        {
            sum += *input++ * *weight++;
        }
        for (; feedback < feedbackEnd;)
        {
            sum += *feedback++ * *weight++;
        }

        *output += sum * bias->multiplier;
    }
}

void RecurrentKernelImpl1B2B(RecurrentConfig const * const config)
{
    nn_bias_c const * bias = config->biasesCompound;
    nn_bias_c const * const biasEnd = bias + config->outputElementCount;
    int16_t const * input = config->input;
    int16_t const * const inputEnd = input + config->inputElementCount;
    int8_t * feedback = (int8_t *)config->feedbackBuffer;
    int8_t const * const feedbackEnd = feedback + config->outputElementCount * config->bytesPerOutput;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;
    int32_t sum;

    for (; bias < biasEnd; bias++, output++)
    {
        *output = bias->bias;
        sum = 0;
        input = config->input;
        feedback = (int8_t *)config->feedbackBuffer;

        for (; input < inputEnd;)
        {
            sum += *input++ * *weight++;
        }
        for (; feedback < feedbackEnd;)
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

        *output += sum * bias->multiplier;
    }
}

void RecurrentKernelImpl1B1B(RecurrentConfig const * const config)
{
    int8_t const * bias = (int8_t*)config->biasesSimple;
    int8_t const * const biasEnd = bias + config->outputElementCount * config->bytesPerBias;
    int8_t const * input = (int8_t*)config->input;
    int8_t const * const inputEnd = input + config->inputElementCount;
    int8_t * feedback = (int8_t*)config->feedbackBuffer;
    int8_t const * const feedbackEnd = feedback + config->outputElementCount * config->bytesPerOutput;
    int8_t const * weight = config->weights1B;
    int32_t * output = config->output;

    for (; bias < biasEnd; bias += config->bytesPerBias, output++)
    {
        if (config->bytesPerBias == 1)
            *output = *(int8_t*)bias;
        else if (config->bytesPerBias == 2)
            *output = *(int16_t*)bias;
        else if (config->bytesPerBias == 4)
            *output = *(int32_t*)bias;

        input = (int8_t*)config->input;
        feedback = (int8_t*)config->feedbackBuffer;

        for (; input < inputEnd;)
        {
            *output += *input++ * *weight++;
        }
        for (; feedback < feedbackEnd;)
        {
            if (config->bytesPerOutput == 1)
            {
                *output += *feedback++ * *weight++;
            }
            else if (config->bytesPerOutput == 2)
            {
                *output += *(int16_t*)feedback * *weight++;
                feedback += 2;
            }
        }
    }
}
