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

#include "Address.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Macros.h"
#include "Shape.h"
#include "Tensor.h"

#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <cstdint>
#include <map>

using namespace GNA;

void CnnLayer::ExpectValid() const
{
    Expect::Equal(validator->Operation, INTEL_CONVOLUTIONAL, Gna2StatusXnnErrorLyrOperation);
    Expect::One(Input.Grouping, Gna2StatusXnnErrorGrouping);
    Expect::One(Output.Grouping, Gna2StatusXnnErrorGrouping);
}

std::unique_ptr<const PoolingFunction> CnnLayer::GetPooling(const nn_layer& layer) const
{
    return PoolingFunction::Create(layer.pLayerStruct, Convolution->Output, *validator, Input.Mode);
}

std::unique_ptr<const PoolingFunction> CnnLayer::GetPooling(const Gna2Operation& apiOperation) const
{
    return PoolingFunction::Create(apiOperation, Convolution->Output, *validator, Input.Mode);
}

void CnnLayer::Init()
{
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
            Gna2StatusXnnErrorLyrInvalidTensorDimensions);
    }
    else
    {
        Expect::NotNull(Activation.get(), Gna2StatusXnnErrorPwlSegments); // Activation is required for cnn with pooling
        Expect::Equal(Pooling->OutputsPerFilterCount, Output.Count / Convolution->Output.at(GNA_DIM_D), Gna2StatusXnnErrorLyrInvalidTensorDimensions);
        // TODO:3: new error

        Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computeHiddenPool(accel, executionConfig); };

        Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
        {this->computePool(layerConfiguration, accel, executionConfig); };
    }
}

Tensor const & CnnLayer::GetOperand(uint32_t operandIndex) const
{
    // TODO:3:replace with generic solution when all layers are transforms
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        return Output.ScratchPad;
    case FilterOperandIndex:
        if (Convolution)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Convolution->Filters);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case BiasOperandIndex:
        if (Convolution)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Convolution->Biases);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case PwlOperandIndex:
        if (Activation)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Activation->Segments);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    default:
        return Layer::GetOperand(operandIndex);
    }
}

void CnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    Layer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputOperandIndex) > 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputOperandIndex];
        Input.ValidateBuffer(inputBuffer);
    }

    // TODO:3: simplify, too fancy logic
    BaseAddress filterOutputBuffer = Activation ? Output.ScratchPad :
        (layerConfiguration.Buffers.count(OutputOperandIndex) > 0 ? layerConfiguration.Buffers[OutputOperandIndex] : Output);

    BaseAddress pwlOutputBuffer = layerConfiguration.Buffers.count(OutputOperandIndex) > 0
        ? layerConfiguration.Buffers[OutputOperandIndex]
        : Output;

    if (layerConfiguration.Buffers.count(OutputOperandIndex) > 0)
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
                { Output.ScratchPad, pwlOutputBuffer });
        }
    }
    else
    {
        configs.Convolution = Convolution->GetRequestConfig(inputBuffer, pwlOutputBuffer);
    }
}
const char * enforcingOutputTensorLayout = "GNA1";
std::unique_ptr<GNA::Layer> CnnLayer::CreateEnforced(const Gna2Operation& operation,
    const BaseValidator& validatorIn)
{
    // TODO: GNA2: Enforce without following const cast
    auto & outputTensor = *const_cast<Gna2Tensor*>(operation.Operands[OutputOperandIndex]);
    const auto outputTensorCopy = outputTensor;
    ModelWrapper::SetLayout(outputTensor, enforcingOutputTensorLayout);
    auto enforcedLayer = std::make_unique<CnnLayer>(operation, validatorIn);
    outputTensor = outputTensorCopy;
    return enforcedLayer;
}

bool CnnLayer::IsForced(const Gna2Operation& operation)
{
    return strncmp(operation.Operands[OutputOperandIndex]->Layout, enforcingOutputTensorLayout, 4) == 0;
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
    auto convConfig = ConvolutionConfig{ Convolution->GetHiddenConfig(), execution };

    //TODO:3: Refactor
    convConfig.pooledOutputs = Output.Buffer;

    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}

void CnnLayer::computePool(LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ layerConfiguration.Configs.Convolution.get(), execution };
    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}

DataConfig CnnLayer::GetDataMode() const
{
    return DataConfig(Input.Mode.Value, Convolution->Filters->Mode.Value,
        Convolution->Biases->Mode.Value, Output.Mode.Value);
}

const nn_layer_conv & CnnLayer::getDetails(const nn_layer & cnn1DLayer)
{
    Expect::NotNull(cnn1DLayer.pLayerStruct);
    return *reinterpret_cast<const nn_layer_conv*>(cnn1DLayer.pLayerStruct);
}

const Gna2Operation & CnnLayer::getDetails(const Gna2Operation & operation)
{
    return operation;
}
