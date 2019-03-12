/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "ConvolutionalLayer.h"

#include "Expect.h"
#include "LayerConfiguration.h"
#include "Macros.h"

using namespace GNA;

CnnLayer::CnnLayer(nn_layer const * const layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress())
{
    Expect::One(Input.at(GNA_DIM_N), XNN_ERR_GROUPING);
    Expect::Equal(Input.at(GNA_DIM_N), Output.at(GNA_DIM_N), XNN_ERR_GROUPING);

    // TODO:3: use Input,Output tensor everywhere
    Convolution = ConvolutionFunction::Create(&Input,
        ActivationFunction::IsEnabled(layer) ? &Output.ScratchPad : &Output,
        layer->pLayerStruct, *validator);
    Activation = ActivationFunction::Create({&Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
        layer->pLayerStruct, *validator}),
    Pooling = PoolingFunction::Create(layer->pLayerStruct, Convolution->Output, *validator, Input.Mode);

    if (!Pooling)
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
        Expect::Equal(Convolution->OutputsPerFilterCount, Output.Count / Convolution->Output.at(GNA_DIM_D),
            XNN_ERR_LYR_INVALID_TENSOR_DIMENSIONS);
        // TODO:3: new error
    }
    else
    {
        Expect::NotNull(Activation.get(), XNN_ERR_PWL_SEGMENTS); // Activation is required for cnn with pooling
        Expect::Equal(Pooling->OutputsPerFilterCount, Output.Count / Convolution->Output.at(GNA_DIM_D), XNN_ERR_LYR_INVALID_TENSOR_DIMENSIONS);
        // TODO:3: new error

        Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computeHiddenPool(accel, executionConfig); };

        Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computePool(layerConfiguration, accel, executionConfig); };
    }
}

void CnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    Layer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputComponent))
    {
        inputBuffer = layerConfiguration.Buffers[InputComponent];
        Input.ValidateBuffer(inputBuffer);
    }

    // TODO:3: simplify, too fancy logic
    BaseAddress filterOutputBuffer = Activation ? Output.ScratchPad:
        (layerConfiguration.Buffers.count(OutputComponent) ? layerConfiguration.Buffers[OutputComponent] : Output);

    BaseAddress pwlOutputBuffer = layerConfiguration.Buffers.count(OutputComponent)
        ? layerConfiguration.Buffers[OutputComponent]
        : Output;

    if (layerConfiguration.Buffers.count(OutputComponent))
    {
        if (Activation)
        {
            Output.ValidateBuffer(pwlOutputBuffer);
        }
        else
        {
            Output.ValidateBuffer(filterOutputBuffer);
        }
    }

    auto& configs = layerConfiguration.Configs;
    if (!Pooling)
    {
        configs.Convolution = Convolution->GetRequestConfig(inputBuffer, filterOutputBuffer);
        if (Activation)
        {
            Activation->UpdateConfigBuffers(layerConfiguration.ConfigList,
                {Output.ScratchPad, pwlOutputBuffer});
        }
    }
    else
    {
        configs.Convolution = Convolution->GetRequestConfig(inputBuffer, pwlOutputBuffer);
    }
}

void CnnLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->ComputeHidden(accel, execution);
}

void CnnLayer::computeHiddenPwl(AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->ComputeHidden(accel, execution);
    Activation->Compute(accel, nullptr, execution);
}

void CnnLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->Compute(layerConfiguration.Configs.Convolution.get(), accel, execution);
}

void CnnLayer::computePwl(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->Compute(layerConfiguration.Configs.Convolution.get(), accel, execution);
    Activation->Compute(accel, &layerConfiguration, execution);
}

void CnnLayer::computeHiddenPool(AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{Convolution->GetHiddenConfig(), execution};

    //TODO:3: Refactor
    convConfig.pooledOutputs = Output.Buffer;

    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}

void CnnLayer::computePool(LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{layerConfiguration.Configs.Convolution.get(), execution};
    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}
