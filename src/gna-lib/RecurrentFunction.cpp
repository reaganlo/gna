/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "RecurrentFunction.h"

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
#include "AffineLayerCapabilities.h"
#include "Bias.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "LayerConfiguration.h"
#include "OperationConfig.h"
#include "Shape.h"
#include "Validator.h"
#include "Weight.h"

#include "gna2-common-api.h"

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <cstdint>
#include <memory>

using namespace GNA;

const FullCapabilitiesMap RecurrentFunction::outputCapabilities =
{
    {INTEL_RECURRENT, {
        AffineLayerCapabilities::GetOperands(OutputOperandIndex).at(INTEL_RECURRENT)
    }}
};

void RecurrentFunction::ValidateActivation(const Gna2Tensor& activationTensor)
{
    const std::function<void()> command = [&]()
    {
        auto pwlShape = Shape::Create(activationTensor.Shape, GNA_TENSOR_N);
        pwlShape.ExpectFits({ GNA_TENSOR_N, XNN_N_PWL_SEGS_MAX });
        ModelErrorHelper::ExpectNotNull(activationTensor.Data);
    };
    ModelErrorHelper::ExecuteForModelItem(command, PwlOperandIndex);
}

// Could not split into separate methods for each component as multibias weight scaling is using bias' and weights; tensors...
std::unique_ptr<RecurrentFunction> RecurrentFunction::Create(
    const TransformFactoryConfig& config,
    const OperationConfig& operationConfig)
{
    auto delay = operationConfig.FeedbackDelay;
    auto activationTensor = config.GetActivation();

    // TODO: 3: remove when ActivationFunction created before PwlCached
    ValidateActivation(activationTensor);

    auto pwlCached = std::make_unique<const PwlCached>(config.outputMode, reinterpret_cast<nn_pwl_seg *>(activationTensor.Data),
        activationTensor.Shape.Dimensions[0]);
    auto kernelOperation = operationConfig.GetKernelOperation();
    auto weightTensor = operationConfig.WeightsTensor;
    auto biasTensor = operationConfig.BiasesTensor;
    auto weights = std::make_unique<const WeightTensor>(weightTensor, config.validator);
    auto biases = std::make_unique<const BiasTensor>(
        biasTensor, 0, Gna2BiasModeDefault, config.validator);
    auto kernelMode = KernelMode{ config.input->Mode, weights->Mode, biases->Mode };
    const auto& affineKernel = AccelerationDetector::GetKernelMap<RecurrentKernel>(
        static_cast<kernel_op>(kernelOperation), kernelMode);

    auto recurrentFunction = std::make_unique<RecurrentFunction>(
        BaseTransformConfig<RecurrentKernel>{config, affineKernel},
        std::move(pwlCached), operationConfig.GetTransformOperation(),
        delay, std::move(weights), std::move(biases));

    //TODO:3:Simplify to make copying not needed
    auto configCopy = config;
    configCopy.input = recurrentFunction->Output.get();
    auto activation = ActivationFunction::Create(configCopy);
    Expect::NotNull(activation.get(), Gna2StatusModelConfigurationInvalid);
    recurrentFunction->SetActivationFunction(std::move(activation));
    return recurrentFunction;
}

void RecurrentFunction::ValidateFeedbackDelay() const
{
    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectAboveEq(FeedbackDelay, ui32_1);
        ModelErrorHelper::ExpectBelowEq(FeedbackDelay, Input->Dimensions.at('H'));
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, 0);
}

RecurrentFunction::RecurrentFunction(
    const BaseTransformConfig<RecurrentKernel>& config,
    std::unique_ptr<const PwlCached> pwlIn,
    TransformOperation transform, uint32_t delay,
    std::unique_ptr<const WeightTensor> weights,
    std::unique_ptr<const BiasTensor> biases) :
    Transform{ transform, &config.kernels, config.input },
    Weights{ std::move(weights) },
    Biases{ std::move(biases) },
    FeedbackDelay{ delay },
    pwl{ std::move(pwlIn) },
    activationConfig{ config.output->Dimensions.at('W'), pwl.get() }
{
    ValidateFeedbackDelay();

    Output = std::make_unique<Tensor>(
        Shape{ GNA_TENSOR_HW, config.output->Dimensions.at('H'), config.output->Dimensions.at('W') },
        config.output->Mode, config.outputBuffer,
        Validator{ config.validator, outputCapabilities });
    Expect::Equal(Input->Dimensions.at('H'), Output->Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);

    auto feedbackBuffer = CalculateFeedbackBuffer(config.output->Buffer);
    auto kernelRecurrentConfig = RecurrentConfig{ config.output->Dimensions.at('W'),
        config.input->Dimensions.at('H'), config.input->Dimensions.at('W'),
        config.input->Buffer, feedbackBuffer, config.outputBuffer, config.output->Buffer,
        *Weights, *Biases, Biases->Mode.Size, config.output->Mode.Size, activationConfig };

    hiddenConfig = std::make_unique<KernelConfig<RecurrentConfig>>(kernelRecurrentConfig,
        BaseConfig{ Input->Buffer, Output->Buffer });
}

void RecurrentFunction::UpdateConfigBuffers(
    std::unique_ptr<BaseConfig> configs[TransformOperationCount],
    const BufferMap& buffers) const
{
    Transform::UpdateConfigBuffers(configs, buffers);

    if (buffers.count(OutputOperandIndex) != 0)
    {
        auto config = GetConfig(configs);
        config->Transform.feedbackBuffer = CalculateFeedbackBuffer(buffers.at(OutputOperandIndex));
        config->Transform.output = hiddenConfig->Transform.output;
    }
}

const BaseAddress RecurrentFunction::CalculateFeedbackBuffer(const BaseAddress& outputBuffer) const
{
    if (outputBuffer)
    {
        auto delaySize = (FeedbackDelay * Output->Dimensions.at('W') * Output->Mode.Size);
        const auto buffer = outputBuffer - delaySize;

        try
        {
            Output->ValidateBuffer(buffer);
        }
        catch (const GnaException&)
        {
            throw GnaException(Gna2StatusXnnErrorNoFeedback);
        }
        return buffer;
    }

    return BaseAddress();
}

const ActivationFunction & RecurrentFunction::GetActivationFunction() const
{
    Expect::NotNull(Activation.get(), Gna2StatusUnknownError);
    return *Activation;
}

void RecurrentFunction::SetActivationFunction(std::unique_ptr<ActivationFunction> activation)
{
    Activation = std::move(activation);
}

Tensor const& RecurrentFunction::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case WeightOperandIndex:
    {
        return GetOperandIfExistOrThrow(Weights);
    }
    case BiasOperandIndex:
    {
        return GetOperandIfExistOrThrow(Biases);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}
