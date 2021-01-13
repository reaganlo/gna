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

#include "XnnKernel.h"

#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "pwl.h"

#include "KernelArguments.h"
#include "KernelMacros.h"
#include "Macros.h"

#include <cstdint>
#include <stdexcept>

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

void recurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B(config);
        config->RequestConfig->Inputs += 2 * inputElementCount;
        runConfig.feedbackBuffer += outputElementCount;
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

void recurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B(config);
        config->RequestConfig->Inputs += 2 * inputElementCount;
        runConfig.feedbackBuffer += outputElementCount;
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

#if OPT_LEVEL < 2
void recurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B1B(config);
        config->RequestConfig->Inputs += inputElementCount;
        if (config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}
void recurrentKernelImpl1B2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl1B2B(config);
        config->RequestConfig->Inputs += 2 * inputElementCount;
        if (config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

void recurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B1B(config);
        config->RequestConfig->Inputs += inputElementCount;
        if (config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

void recurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> const * const config)
{
    auto& runConfig = config->RequestConfig->Transform;
    auto activationCfg = ExecutionKernelConfig<ActivationConfig>{
        &runConfig.activation, *config};
    auto& activation = activationCfg.RequestConfig->Transform;
    auto io = activationCfg.RequestConfig;
    io->Inputs = reinterpret_cast<int8_t const *>(runConfig.output);
    io->Outputs = config->RequestConfig->Outputs;

    auto feedback = runConfig.feedbackBuffer;
    auto outputs = runConfig.output;
    auto inputs = config->RequestConfig->Inputs;

    auto inputVectorCount = runConfig.inputVectorCount;
    auto inputElementCount = runConfig.inputElementCount;
    auto outputElementCount = runConfig.outputElementCount;

    // for each input vector
    for (uint32_t i = 0; i < inputVectorCount; i++)
    {
        RecurrentKernelImpl2B2B(config);
        config->RequestConfig->Inputs += inputElementCount * 2;
        if(config->RequestConfig->Transform.bytesPerOutput == 1)
        {
            runConfig.feedbackBuffer = (int16_t*)((uint64_t)runConfig.feedbackBuffer
                                        + outputElementCount);
        }
        else
        {
            runConfig.feedbackBuffer += outputElementCount;
        }
        runConfig.output += outputElementCount;

        activation.Kernel->InitializeActivationFunctions();
        activation.Kernel->ActivateAll(&activationCfg);
        io->Inputs = io->Inputs + activation.ElementCount * 4;
        io->Outputs = io->Outputs +
            activation.ElementCount * config->RequestConfig->Transform.bytesPerOutput;
    }

    // restore pointers in config
    runConfig.feedbackBuffer = feedback;
    runConfig.output = outputs;
    config->RequestConfig->Inputs = inputs;
}

#endif
void copyKernelImpl(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * static_cast<uint32_t>(sizeof(int16_t));

    for (row = 0; row < config->rowCount; row++)
    {
        memmove_s(
            config->output + (config->outputColumnCount * row),
            bytesToCopy,
            config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}

void copyKernelImpl1B(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * static_cast<uint32_t>(sizeof(int8_t));

    for (row = 0; row < config->rowCount; row++)
    {
        memmove_s(
            (int8_t*)config->output + (config->outputColumnCount * row),
            bytesToCopy,
            (int8_t*)config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}

void copyKernelImpl2B(CopyConfig const * const config)
{
    uint32_t row;
    uint32_t bytesToCopy = config->columnCount * static_cast<uint32_t>(sizeof(int16_t));

    for (row = 0; row < config->rowCount; row++)
    {
        memmove_s(
            config->output + (config->outputColumnCount * row),
            bytesToCopy,
            config->input + (config->inputColumnCount * row),
            bytesToCopy);
    }
}
#if OPT_LEVEL >=2
#define NotImplementedKernel(kernel) ToUnifiedKernel(CodeCaveMitigationFakeKernel)
static void CodeCaveMitigationFakeKernel()
{
    throw std::logic_error("Call to not defined GNA kernel found!");
}
#endif

template<typename KernelFunctionType>
VoidKernel ToUnifiedKernel(KernelFunctionType kernel)
{
    return reinterpret_cast<VoidKernel>(kernel);
}

template<>
VoidKernel GetXnnKernel<KernelAcceleration, HwConsistencyMode>(KernelType type)
{
    static const VoidKernel Kernels[]=
    {
        ToUnifiedKernel(AffineKernelImpl1B),
        ToUnifiedKernel(AffineKernelImpl2B),

        ToUnifiedKernel(AffineActiveListKernelImpl1B),
        ToUnifiedKernel(AffineActiveListKernelImpl2B),

        ToUnifiedKernel(AffineMultiBiasKernelImpl1B),
        ToUnifiedKernel(AffineMultiBiasKernelImpl2B),

        ToUnifiedKernel(DiagonalKernelImpl1B),
        ToUnifiedKernel(DiagonalKernelImpl2B),

        ToUnifiedKernel(recurrentKernelImpl1B),
        ToUnifiedKernel(recurrentKernelImpl2B),

        ToUnifiedKernel(ConvolutionKernelImpl),
        ToUnifiedKernel(ConvolutionPoolingKernelImpl),

        ToUnifiedKernel(activationKernelImpl),
        ToUnifiedKernel(TransposeKernelImpl),
        ToUnifiedKernel(copyKernelImpl),

    #if OPT_LEVEL < 2

        ToUnifiedKernel(AffineKernelImpl1B1B),
        ToUnifiedKernel(AffineKernelImpl2B1B),
        ToUnifiedKernel(AffineKernelImpl1B2B),
        ToUnifiedKernel(AffineKernelImpl2B2B),
        ToUnifiedKernel(AffineActiveListKernelImpl1B1B),
        ToUnifiedKernel(AffineActiveListKernelImpl2B1B),
        ToUnifiedKernel(AffineActiveListKernelImpl1B2B),
        ToUnifiedKernel(AffineActiveListKernelImpl2B2B),
        ToUnifiedKernel(AffineMultiBiasKernelImpl1B1B),
        ToUnifiedKernel(AffineMultiBiasKernelImpl2B1B),
        ToUnifiedKernel(AffineMultiBiasKernelImpl1B2B),
        ToUnifiedKernel(AffineMultiBiasKernelImpl2B2B),
        ToUnifiedKernel(DiagonalKernelImpl1B1B),
        ToUnifiedKernel(DiagonalKernelImpl2B1B),
        ToUnifiedKernel(DiagonalKernelImpl1B2B),
        ToUnifiedKernel(DiagonalKernelImpl2B2B),
        ToUnifiedKernel(recurrentKernelImpl1B1B),
        ToUnifiedKernel(recurrentKernelImpl2B1B),
        ToUnifiedKernel(recurrentKernelImpl1B2B),
        ToUnifiedKernel(recurrentKernelImpl2B2B),
        ToUnifiedKernel(ConvolutionKernelImpl1B),
        ToUnifiedKernel(ConvolutionPoolingKernelImpl1B),
        ToUnifiedKernel(ConvolutionKernelImpl2B),
        ToUnifiedKernel(ConvolutionPoolingKernelImpl2B),
        ToUnifiedKernel(TransposeKernelImpl1B),
        ToUnifiedKernel(TransposeKernelImpl2B),
        ToUnifiedKernel(copyKernelImpl1B),
        ToUnifiedKernel(copyKernelImpl2B),

        ToUnifiedKernel(Convolution2DKernelImpl1B1B),
        ToUnifiedKernel(Convolution2DKernelImpl1B2B),
        ToUnifiedKernel(Convolution2DKernelImpl2B1B),
        ToUnifiedKernel(Convolution2DKernelImpl2B2B),

        ToUnifiedKernel(Pooling2DKernelImpl1B),
        ToUnifiedKernel(Pooling2DKernelImpl2B),
        ToUnifiedKernel(Pooling2DKernelImpl4B),
    #else
        NotImplementedKernel(AffineKernelImpl1B1B),
        NotImplementedKernel(AffineKernelImpl2B1B),
        NotImplementedKernel(AffineKernelImpl1B2B),
        NotImplementedKernel(AffineKernelImpl2B2B),
        NotImplementedKernel(AffineActiveListKernelImpl1B1B),
        NotImplementedKernel(AffineActiveListKernelImpl2B1B),
        NotImplementedKernel(AffineActiveListKernelImpl1B2B),
        NotImplementedKernel(AffineActiveListKernelImpl2B2B),
        NotImplementedKernel(AffineMultiBiasKernelImpl1B1B),
        NotImplementedKernel(AffineMultiBiasKernelImpl2B1B),
        NotImplementedKernel(AffineMultiBiasKernelImpl1B2B),
        NotImplementedKernel(AffineMultiBiasKernelImpl2B2B),
        NotImplementedKernel(DiagonalKernelImpl1B1B),
        NotImplementedKernel(DiagonalKernelImpl2B1B),
        NotImplementedKernel(DiagonalKernelImpl1B2B),
        NotImplementedKernel(DiagonalKernelImpl2B2B),
        NotImplementedKernel(recurrentKernelImpl1B1B),
        NotImplementedKernel(recurrentKernelImpl2B1B),
        NotImplementedKernel(recurrentKernelImpl1B2B),
        NotImplementedKernel(recurrentKernelImpl2B2B),
        NotImplementedKernel(ConvolutionKernelImpl1B),
        NotImplementedKernel(ConvolutionPoolingKernelImpl1B),
        NotImplementedKernel(ConvolutionKernelImpl2B),
        NotImplementedKernel(ConvolutionPoolingKernelImpl2B),
        NotImplementedKernel(TransposeKernelImpl1B),
        NotImplementedKernel(TransposeKernelImpl2B),
        NotImplementedKernel(copyKernelImpl1B),
        NotImplementedKernel(copyKernelImpl2B),
        NotImplementedKernel(Convolution2DKernelImpl1B1B),
        NotImplementedKernel(Convolution2DKernelImpl1B2B),
        NotImplementedKernel(Convolution2DKernelImpl2B1B),
        NotImplementedKernel(Convolution2DKernelImpl2B2B),
        NotImplementedKernel(Pooling2DKernelImpl1B),
        NotImplementedKernel(Pooling2DKernelImpl2B),
        NotImplementedKernel(Pooling2DKernelImpl4B),
    #endif
    };
    return Kernels[type];
}

}
