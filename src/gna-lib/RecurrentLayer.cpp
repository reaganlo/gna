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
#include "Validator.h"

using namespace GNA;

RnnLayer::RnnLayer(nn_layer const * const layer) :
    Layer(layer),
    Affine{AffineFunction::Create(&static_cast<const nn_layer_reccurent*>(layer->pLayerStruct)->affine)},
    // RNN has only 2B output with Activation always enabled
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_reccurent*>(layer->pLayerStruct)->pwl, true,
        Output.ScratchPad, PwlOutputConfig{})),
    FeedbackDelay{static_cast<const nn_layer_reccurent * const>(layer->pLayerStruct)->feedbackFrameDelay},
    recurrentKernels{AccelerationDetector::GetKernelMap<RecurrentKernel>(Affine->GetWeightMode())},
    rnnHiddenConfig{Output.ElementCount, Input.VectorCount, Input.ElementCount, Input.Buffer, nullptr,
                        Output.ScratchPad, Output.Buffer, Affine->GetWeights(), Affine->GetBiases()}
{
    Expect::InRange(FeedbackDelay, 1, Input.VectorCount - 1, XNN_ERR_NO_FEEDBACK);

    // must be multiple 32 to keep 64B output buffer alignment
    Expect::MultiplicityOf(Output.ElementCount, RNN_N_OUT_ELEMS_MPLY);
    Output.SetOutputMode(Activation.operator bool(), layer->nBytesPerOutput);

    Expect::True(Input.VectorCount == Output.VectorCount, XNN_ERR_LYR_CFG);

    if (INTEL_INPUT == Config.Type || INTEL_HIDDEN == Config.Type)
    {
        feedbackBuffer = CalculateFeedbackBuffer(Output.Buffer);
        rnnHiddenConfig.feedbackBuffer = feedbackBuffer;
    }

    Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeHidden(accel, fvBuffers, saturationCount); };

    Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) 
                    {this->computeConfig(layerConfiguration, accel, fvBuffers, saturationCount); };
}

void RnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    auto outputBuffer = layerConfiguration.OutputBuffer
        ? layerConfiguration.OutputBuffer->Get<int32_t>() : Output.Buffer;

    auto& configs = layerConfiguration.Configs;

    if(!configs.Recurrent)
        configs.Recurrent = std::make_unique<RecurrentConfig>(rnnHiddenConfig);
    configs.Recurrent->input = inputBuffer;
    configs.Recurrent->output = outputBuffer;

    if(outputBuffer)
        configs.Recurrent->feedbackBuffer = CalculateFeedbackBuffer(outputBuffer);
}

void RnnLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto rnnConfig = RecurrentConfig{&rnnHiddenConfig, saturationCount};

    recurrentKernels.at(accel)(&rnnConfig, &Activation->Pwl);
}

void RnnLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount) const
{
    auto rnnConfig = RecurrentConfig{layerConfiguration.Configs.Recurrent.get(), saturationCount};
    
    recurrentKernels.at(accel)(&rnnConfig, &Activation->Pwl);
}

const OutputBuffer RnnLayer::CalculateFeedbackBuffer(const OutputBuffer& outputBuffer) const
{
    const auto buffer = outputBuffer - (FeedbackDelay * Output.ElementCount);
    Expect::ValidBuffer(buffer, XNN_ERR_NO_FEEDBACK);
    return buffer;
}
