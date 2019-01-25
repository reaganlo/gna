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
#include "LayerConfiguration.h"
#include "Expect.h"

using namespace GNA;

RnnLayer::RnnLayer(nn_layer const * const layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, validatorIn),
    FeedbackDelay{static_cast<const nn_layer_reccurent * const>(layer->pLayerStruct)->feedbackFrameDelay},
    recurrentKernels{ AccelerationDetector::GetKernelMap<RecurrentKernel>(
        KERNEL_RECURRENT, {Input.Mode, Affine->Weights->Mode, Affine->Biases->Mode}) },
    rnnHiddenConfig{Output.at(GNA_DIM_H), Input.at(GNA_DIM_N), Input.at(GNA_DIM_W), Input.Buffer, nullptr,
                        Activation->Input->Buffer, Activation->Output->Buffer, *Affine->Weights,
                    *Affine->Biases, Affine->Biases->Mode.Size, Output.Mode.Size, {Output.at(GNA_DIM_H), &Activation->Pwl}}
{
    // TODO:3: think of validation functor for this kind of properties or other means to generalize/unify
    Expect::InRange(FeedbackDelay, ui32_1, Input.at(GNA_DIM_N), XNN_ERR_NO_FEEDBACK);
    Expect::Equal(Input.at(GNA_DIM_N), Output.at(GNA_DIM_N), XNN_ERR_LYR_CFG);

    rnnHiddenConfig.feedbackBuffer = CalculateFeedbackBuffer(Output);

    Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

void RnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = layerConfiguration.Buffers.count(InputComponent)
        ? layerConfiguration.Buffers[InputComponent] : Input;

    BaseAddress outputBuffer = layerConfiguration.Buffers.count(OutputComponent)
        ? layerConfiguration.Buffers[OutputComponent] : Output;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Recurrent)
        configs.Recurrent = std::make_unique<RecurrentConfig>(rnnHiddenConfig);
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
        const auto buffer = outputBuffer - (FeedbackDelay * Output.at(GNA_DIM_H) * Output.Mode);

        try
        {
            Output.ValidateBuffer(buffer);
        }
        catch (const GnaException&)
        {
            throw GnaException(XNN_ERR_NO_FEEDBACK);
        }
        return buffer;
    }
    else
    {
        return BaseAddress();
    }
}
