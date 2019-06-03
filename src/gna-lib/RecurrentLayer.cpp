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

#include "RecurrentLayer.h"

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "Bias.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "Layer.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Tensor.h"
#include "Weight.h"

#include "gna-api.h"
#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <memory>

namespace GNA
{
class BaseValidator;
}

using namespace GNA;

RnnLayer::RnnLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, validatorIn),
    FeedbackDelay{static_cast<const nn_layer_recurrent *>(layer.pLayerStruct)->feedbackFrameDelay},
    recurrentKernels{ AccelerationDetector::GetKernelMap<RecurrentKernel>(
        KERNEL_RECURRENT, {Input.Mode, Affine->Weights->Mode, Affine->Biases->Mode}) },
    rnnHiddenConfig{Output.Dimensions.at('W'), Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, nullptr,
                        Activation->Input->Buffer, Activation->Output->Buffer, *Affine->Weights,
                    *Affine->Biases, Affine->Biases->Mode.Size, Output.Mode.Size, {Output.Dimensions.at('W'), &Activation->Pwl}}
{
    // TODO:3: think of validation functor for this kind of properties or other means to generalize/unify
    Expect::InRange(FeedbackDelay, ui32_1, Input.Dimensions.at('H'), Gna2StatusXnnErrorNoFeedback);
    Expect::Equal(Input.Dimensions.at('H'), Output.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);

    rnnHiddenConfig.feedbackBuffer = CalculateFeedbackBuffer(Output);

    Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

RnnLayer::RnnLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    AffineBaseLayer(operation, validatorIn),
    FeedbackDelay{*static_cast<uint32_t *>(operation.Parameters[0])},
    recurrentKernels{ AccelerationDetector::GetKernelMap<RecurrentKernel>(
        KERNEL_RECURRENT, {Input.Mode, Affine->Weights->Mode, Affine->Biases->Mode}) },
    rnnHiddenConfig{Output.Dimensions.at('W'), Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, nullptr,
                        Activation->Input->Buffer, Activation->Output->Buffer, *Affine->Weights,
                    *Affine->Biases, Affine->Biases->Mode.Size, Output.Mode.Size, {Output.Dimensions.at('W'), &Activation->Pwl}}
{
    // TODO:3: think of validation functor for this kind of properties or other means to generalize/unify
    Expect::InRange(FeedbackDelay, ui32_1, Input.Dimensions.at('H'), Gna2StatusXnnErrorNoFeedback);
    Expect::Equal(Input.Dimensions.at('H'), Output.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);

    rnnHiddenConfig.feedbackBuffer = CalculateFeedbackBuffer(Output);

    Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

void RnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = layerConfiguration.Buffers.count(InputComponent) > 0
        ? layerConfiguration.Buffers[InputComponent] : Input;

    BaseAddress outputBuffer = layerConfiguration.Buffers.count(OutputComponent) > 0
        ? layerConfiguration.Buffers[OutputComponent] : Output;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Recurrent)
    {
        configs.Recurrent = std::make_unique<RecurrentConfig>(rnnHiddenConfig);
    }
    configs.Recurrent->input = inputBuffer;
    Input.ValidateBuffer(inputBuffer);

    if (outputBuffer)
    {
        configs.Recurrent->feedbackBuffer = CalculateFeedbackBuffer(outputBuffer);
        configs.Recurrent->activation.Update({{OutputComponent, outputBuffer}});
    }
}

void RnnLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    auto rnnConfig = RecurrentConfig{&rnnHiddenConfig, executionConfig};

    recurrentKernels.at(accel)(&rnnConfig);
}

void RnnLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    auto rnnConfig = RecurrentConfig{layerConfiguration.Configs.Recurrent.get(), executionConfig};

    recurrentKernels.at(accel)(&rnnConfig);
}

const BaseAddress RnnLayer::CalculateFeedbackBuffer(const BaseAddress& outputBuffer) const
{
    if (outputBuffer)
    {
        auto delaySize = (FeedbackDelay * Output.Dimensions.at('W') * Output.Mode.Size);
        const auto buffer = outputBuffer - delaySize;

        try
        {
            Output.ValidateBuffer(buffer);
        }
        catch (const GnaException&)
        {
            throw GnaException(Gna2StatusXnnErrorNoFeedback);
        }
        return buffer;
    }

    return BaseAddress();
}
