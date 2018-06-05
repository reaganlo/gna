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

#include "common.h"
#include "XnnKernelApi.h"

#include <string.h>

#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "KernelMacros.h"

namespace GNA
{

#define pwlKernelImpl KERNEL(pwlKernelImpl)
#define recurrentKernelImpl1B KERNEL(recurrentKernelImpl1B)
#define recurrentKernelImpl2B KERNEL(recurrentKernelImpl2B)
#define copyKernelImpl KERNEL(copyKernelImpl)

void pwlKernelImpl(PwlCached const * const pwl, PwlOutputConfig const * const outputConfig)
{
    pwl->ActivateAll(&pwl->pwl, outputConfig);
}

void recurrentKernelImpl1B(RecurrentConfig const * const config, PwlCached const * const pwl)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto runPwlOutputConfig = PwlOutputConfig(runConfig.pwlOutputConfig);

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl1B(&runConfig);
        runConfig.input += config->inputElementCount;
        runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        pwl->ActivateAll(&pwl->pwl, &runPwlOutputConfig);
        runPwlOutputConfig.input += runPwlOutputConfig.elementCount;
        runPwlOutputConfig.output += runPwlOutputConfig.elementCount;
    }
}

void recurrentKernelImpl2B(RecurrentConfig const * const config, PwlCached const * const pwl)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto runPwlOutputConfig = PwlOutputConfig(runConfig.pwlOutputConfig);

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl2B(&runConfig);
        runConfig.input += config->inputElementCount;
        runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        pwl->ActivateAll(&pwl->pwl, &runPwlOutputConfig);
        runPwlOutputConfig.input += runPwlOutputConfig.elementCount;
        runPwlOutputConfig.output += runPwlOutputConfig.elementCount;
    }
}

void copyKernelImpl(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * sizeof(int16_t);

    for (row = 0; row < config->rowCount; row++)
    {
        memcpy_s(
            config->output + (config->outputColumnCount * row),
            bytesToCopy,
            config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}

XnnKernel KERNEL(xnnKernel) =
{
    AffineKernelImpl1B,
    AffineKernelImpl2B,
    AffineActiveListKernelImpl1B,
    AffineActiveListKernelImpl2B,

    AffineMultiBiasKernelImpl1B,
    AffineMultiBiasKernelImpl2B,

    DiagonalKernelImpl1B,
    DiagonalKernelImpl2B,

    recurrentKernelImpl1B,
    recurrentKernelImpl2B,

    ConvolutionKernelImpl,
    ConvolutionPoolingKernelImpl,

    pwlKernelImpl,
    TransposeKernelImpl,
    copyKernelImpl,
};

}
