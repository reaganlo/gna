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

#include "ActivationHelper.h"
#include "ActiveList.h"
#include "Address.h"
#include "Bias.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Tensor.h"
#include "Weight.h"

#include "gna2-common-api.h"

#include "common.h"
#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <algorithm>
#include <map>

using namespace GNA;

AffineBaseLayer::AffineBaseLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    Affine(AffineFunction::Create(Input,
        ActivationHelper::IsEnabled(layer) ? Output.ScratchPad : Output,
        layer.pLayerStruct, *validator)),
    // TODO:3: refactor to Transform and to use Affine->Output
    Activation(ActivationFunction::Create({&Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
        layer, *validator}))

{
    initComputeFunctions();
}

AffineBaseLayer::AffineBaseLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    Layer(operation, validatorIn, {}, BaseAddress()),
    Affine(AffineFunction::Create(Input, ActivationHelper::IsEnabled(operation)
            ? Output.ScratchPad : Output, operation, *validator)),
    // TODO:3: refactor to Transform and to use Affine->Output
    Activation(ActivationFunction::Create({&Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
        operation, *validator}))

{
    initComputeFunctions();
}

void AffineBaseLayer::initComputeFunctions()
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

Tensor const & AffineBaseLayer::GetOperand(uint32_t operandIndex) const
{
    // TODO:3:replace with generic solution when all layers are transforms
    switch (operandIndex)
    {
    case 2:
        if (Affine)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Affine->Weights);
        }
    case 3:
    if (Affine)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Affine->Biases);
        }
    case 4:
        if (Activation)
        {
            return Activation->GetOperand(2);// TODO:3:Intentional literal, replace with generic solution when all layers are transforms
        }
    /*case 5:
        if (Affine && Affine->WeightScaleFactors)
        {
            return *Affine->WeightScaleFactors;
        }*/
    default:
        return Layer::GetOperand(operandIndex);
    }
}

void AffineBaseLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    Layer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent) > 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputComponent) > 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputComponent];
        Output.ValidateBuffer(layerConfiguration.Buffers[OutputComponent]);
    }

    auto& configs = layerConfiguration.Configs;

    if (Activation)
    {
        configs.Affine = Affine->GetRequestConfig(inputBuffer, Output.ScratchPad);
        if (outputBuffer)
        {
            Activation->UpdateConfigBuffers(layerConfiguration.ConfigList,
                {{OutputComponent, outputBuffer}});
        }
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

AffineLayer::AffineLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, validatorIn)
{}

AffineLayer::AffineLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    AffineBaseLayer(operation, validatorIn)
{}

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);
    if (Activation)
    {
        if (layerConfiguration.ActList)
        {
            Expect::InRange(layerConfiguration.ActList->IndicesCount, ui32_1, Output.Dimensions.at('H'), Gna2StatusActiveListIndicesInvalid);
        }
        auto const outputCount = layerConfiguration.ActList ?
            layerConfiguration.ActList->IndicesCount : Output.Dimensions.at('H');
        Activation->UpdateActiveOutputCount(layerConfiguration.ConfigList, outputCount * Output.Dimensions.at('W'));
    }
}

AffineDiagonalLayer::AffineDiagonalLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, validatorIn)
{
    Expect::Equal(Input.Dimensions.at('H'), Output.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);
    Expect::Equal(Input.Dimensions.at('W'), Output.Dimensions.at('W'), Gna2StatusXnnErrorLyrCfg);
}
