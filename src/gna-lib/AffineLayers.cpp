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

#include "AffineLayers.h"

#include "common.h"
#include "LayerConfiguration.h"
#include "Expect.h"

using std::make_unique;

using namespace GNA;

AffineBaseLayer::AffineBaseLayer(const nn_layer *layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    Affine(AffineFunction::Create(&Input,
        ActivationFunction::IsEnabled(layer) ? &Output.ScratchPad : &Output,
        layer->pLayerStruct, *validator)),
    // TODO:3: refactor to Transform and to use Affine->Output
    Activation(ActivationFunction::Create({&Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
        layer->pLayerStruct, *validator}))

{
    if (Activation)
    {
        Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computeHiddenPwl(accel, executionConfig); };

        Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computePwl(layerConfiguration, accel, executionConfig); };
    }
    else
    {
        Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computeHidden(accel, executionConfig); };

        Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->compute(layerConfiguration, accel, executionConfig); };
    }
}

void AffineBaseLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    Layer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent))
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputComponent))
    {
        outputBuffer = layerConfiguration.Buffers[OutputComponent];
        Output.ValidateBuffer(layerConfiguration.Buffers[OutputComponent]);
    }

    auto& configs = layerConfiguration.Configs;

    if (Activation)
    {
        configs.Affine = Affine->GetRequestConfig(inputBuffer, Output.ScratchPad);
        if (outputBuffer)
            Activation->UpdateConfigBuffers(layerConfiguration.ConfigList,
                {{OutputComponent, outputBuffer}});
    }
    else
    {
        configs.Affine = Affine->GetRequestConfig(inputBuffer, outputBuffer);
    }
}

void AffineBaseLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & execution) const
{
    Affine->ComputeHidden(accel, execution);
}

void AffineBaseLayer::computeHiddenPwl(AccelerationMode accel, ExecutionConfig const & execution) const
{
    Affine->ComputeHidden(accel, execution);

    Activation->Compute(accel, nullptr, execution);
}

void AffineBaseLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    Affine->Compute(layerConfiguration, accel, execution);
}

void AffineBaseLayer::computePwl(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    Affine->Compute(layerConfiguration, accel, execution);

    Activation->Compute(accel, &layerConfiguration, execution);
}

DataConfig AffineBaseLayer::GetDataMode() const
{
    auto weightMode = this->Affine->Weights->Mode.Value;
    auto biasMode = this->Affine->Biases->Mode.Value;
    return DataConfig(Input.Mode, weightMode, biasMode, Output.Mode);
}

AffineLayer::AffineLayer(const nn_layer *layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, validatorIn)
{};

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);
    if (Activation)
    {
        if (layerConfiguration.ActList)
        {
            Expect::InRange(layerConfiguration.ActList->IndicesCount, ui32_1, Output.at(GNA_DIM_H), GNA_INVALIDINDICES);
        }
        auto const outputCount = layerConfiguration.ActList ?
            layerConfiguration.ActList->IndicesCount : Output.at(GNA_DIM_H);
        Activation->UpdateActiveOutputCount(layerConfiguration.ConfigList, outputCount * Output.at(GNA_DIM_N));
    }
}

AffineDiagonalLayer::AffineDiagonalLayer(const nn_layer *layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, validatorIn)
{
    Expect::Equal(Input.at(GNA_DIM_W), Output.at(GNA_DIM_H), XNN_ERR_LYR_CFG);
    Expect::Equal(Input.at(GNA_DIM_N), Output.at(GNA_DIM_N), XNN_ERR_LYR_CFG);
}
