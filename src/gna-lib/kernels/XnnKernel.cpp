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
#include "XnnKernel.h"

#include <string.h>

#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "KernelMacros.h"
#include "Macros.h"

namespace GNA
{

#define activationKernelImpl KERNEL(activationKernelImpl)
#define recurrentKernelImpl1B KERNEL(recurrentKernelImpl1B)
#define recurrentKernelImpl2B KERNEL(recurrentKernelImpl2B)
#define copyKernelImpl KERNEL(copyKernelImpl)
#define copyKernelImpl1B KERNEL(copyKernelImpl1B)
#define copyKernelImpl2B KERNEL(copyKernelImpl2B)
#define InitializeActivationFunctions KERNEL(InitializeActivationFunctions)

#if OPT_LEVEL < 2
#define recurrentKernelImpl1B1B KERNEL(recurrentKernelImpl1B1B)
#define recurrentKernelImpl1B2B KERNEL(recurrentKernelImpl1B2B)
#define recurrentKernelImpl2B1B KERNEL(recurrentKernelImpl2B1B)
#define recurrentKernelImpl2B2B KERNEL(recurrentKernelImpl2B2B)
#endif

void activationKernelImpl(ExecutionKernelConfig<ActivationConfig> const * const config)
{
    config->RequestConfig->Transform.Kernel->InitializeActivationFunctions();
    config->RequestConfig->Transform.Kernel->ActivateAll(config);
}

void recurrentKernelImpl1B(RecurrentConfig const * const config)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *runConfig.execution};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl1B(&runConfig);
        runConfig.input += config->inputElementCount;
        runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Update(BufferMap{io->Inputs + activation.ElementCount * 4,
            io->Outputs + activation.ElementCount * config->bytesPerOutput});
    }
}

void recurrentKernelImpl2B(RecurrentConfig const * const config)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *runConfig.execution};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl2B(&runConfig);
        runConfig.input += config->inputElementCount;
        runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Update(BufferMap{io->Inputs + activation.ElementCount * 4,
            io->Outputs + activation.ElementCount * config->bytesPerOutput});
    }
}

#if OPT_LEVEL < 2
void recurrentKernelImpl1B1B(RecurrentConfig const * const config)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *runConfig.execution};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl1B1B(&runConfig);
        runConfig.input = (int16_t*)((uint64_t)runConfig.input + config->inputElementCount);
        if (config->bytesPerOutput == 1)
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer + config->outputElementCount);
        else
            runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Update(BufferMap{io->Inputs + activation.ElementCount * 4,
            io->Outputs + activation.ElementCount * config->bytesPerOutput});
    }
}
void recurrentKernelImpl1B2B(RecurrentConfig const * const config)
{
     auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *runConfig.execution};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl1B2B(&runConfig);
        runConfig.input += config->inputElementCount;
        if (config->bytesPerOutput == 1)
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer + config->outputElementCount);
        else
            runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Update(BufferMap{io->Inputs + activation.ElementCount * 4,
            io->Outputs + activation.ElementCount * config->bytesPerOutput});
    }
}

void recurrentKernelImpl2B1B(RecurrentConfig const * const config)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *runConfig.execution};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl2B1B(&runConfig);
        runConfig.input = (int16_t*)((uint64_t)runConfig.input + config->inputElementCount);
        if (config->bytesPerOutput == 1)
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer + config->outputElementCount);
        else
            runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Update(BufferMap{io->Inputs + activation.ElementCount * 4,
            io->Outputs + activation.ElementCount * config->bytesPerOutput});
    }
}

void recurrentKernelImpl2B2B(RecurrentConfig const * const config)
{
    auto runConfig = RecurrentConfig(*config); // local modifiable copy
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *runConfig.execution};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;

    // for each input vector
    for (uint32_t i = 0; i < config->inputVectorCount; i++)
    {
        RecurrentKernelImpl2B2B(&runConfig);
        runConfig.input += config->inputElementCount;
        if(config->bytesPerOutput == 1)
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer + config->outputElementCount);
        else
            runConfig.feedbackBuffer += config->outputElementCount;
        runConfig.output += config->outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Update(BufferMap{io->Inputs + activation.ElementCount * 4,
            io->Outputs + activation.ElementCount * config->bytesPerOutput});
    }
}
#endif
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

void copyKernelImpl1B(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * sizeof(int8_t);

    for (row = 0; row < config->rowCount; row++)
    {
        memcpy_s(
            (int8_t*)config->output + (config->outputColumnCount * row),
            bytesToCopy,
            (int8_t*)config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}

void copyKernelImpl2B(CopyConfig const * const config)
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

    activationKernelImpl,
    TransposeKernelImpl,
    copyKernelImpl,

#if OPT_LEVEL < 2

    AffineKernelImpl1B1B,
    AffineKernelImpl2B1B,
    AffineKernelImpl1B2B,
    AffineKernelImpl2B2B,
    AffineActiveListKernelImpl1B1B,
    AffineActiveListKernelImpl2B1B,
    AffineActiveListKernelImpl1B2B,
    AffineActiveListKernelImpl2B2B,
    AffineMultiBiasKernelImpl1B1B,
    AffineMultiBiasKernelImpl2B1B,
    AffineMultiBiasKernelImpl1B2B,
    AffineMultiBiasKernelImpl2B2B,
    DiagonalKernelImpl1B1B,
    DiagonalKernelImpl2B1B,
    DiagonalKernelImpl1B2B,
    DiagonalKernelImpl2B2B,
    recurrentKernelImpl1B1B,
    recurrentKernelImpl2B1B,
    recurrentKernelImpl1B2B,
    recurrentKernelImpl2B2B,
    ConvolutionKernelImpl1B,
    ConvolutionPoolingKernelImpl1B,
    ConvolutionKernelImpl2B,
    ConvolutionPoolingKernelImpl2B,
    TransposeKernelImpl1B,
    TransposeKernelImpl2B,
    copyKernelImpl1B,
    copyKernelImpl2B,

    Convolution2DKernelImpl1B1B,
    Convolution2DKernelImpl1B2B,
    Convolution2DKernelImpl2B1B,
    Convolution2DKernelImpl2B2B,

    Pooling2DKernelImpl1B,
    Pooling2DKernelImpl2B,
    Pooling2DKernelImpl4B
#endif
};

}
