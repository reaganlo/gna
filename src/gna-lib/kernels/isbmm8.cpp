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

void DiagonalKernelImpl1B(AffineConfig const * const config)
{
    uint32_t i, j;
    int64_t sum;
    int64_t weightValue;
    int8_t const * weight = config->weights1B;
    int16_t const * input = config->input;
    int32_t * output = config->output;
    nn_bias_c const * bias = config->biasesCompound;

    for (i = 0; i < config->outputElementCount; i++)
    {
        weightValue = bias[i].multiplier * weight[i];
        for (j = 0; j < config->inputVectorCount; j++)
        {
            sum =  bias[i].bias + (weightValue * input[i * config->inputVectorCount + j]);
#if GNA_SAT == 1
            saturate_store_out(&sum, &output[i * config->inputVectorCount + j], config->saturationCount);
#else 
            output[i * config->inputVectorCount + j] = (int32_t)sum;
#endif
        }
    }
}
