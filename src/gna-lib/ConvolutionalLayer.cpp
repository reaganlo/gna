/*
 INTEL CONFIDENTIAL
 Copyright 2018-2020 Intel Corporation.

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


#include <algorithm>
#include <cstdint>
#include <map>

using namespace GNA;

CnnLayer::CnnLayer(const ApiOperation& apiLayer, const BaseValidator& validatorIn) :
    Layer(apiLayer, validatorIn, {}, BaseAddress())
{
    ExpectValid();
    Convolution = GetConvolution(getDetails(apiLayer));
    Activation = ActivationFunction::Create({ &Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
        apiLayer, *validator });
    Pooling = GetPooling(apiLayer);
    Init();
    dataConfig = DataConfig{ Input.Mode, Convolution->Filters->Mode,
        Convolution->Biases->Mode, Output.Mode, Activation == nullptr };
}

void CnnLayer::ExpectValid() const
{
    Expect::Equal(validator->Operation, INTEL_CONVOLUTIONAL, Gna2StatusXnnErrorLyrOperation);
    Expect::One(Input.Grouping, Gna2StatusXnnErrorGrouping);
    Expect::One(Output.Grouping, Gna2StatusXnnErrorGrouping);
}

std::unique_ptr<const PoolingFunction> CnnLayer::GetPooling(const Gna2Operation& apiOperation) const
{
    return PoolingFunction::Create(apiOperation, Convolution->Output, *validator, Input.Mode);
}

void CnnLayer::Init()
{
    uint32_t outputsPerFilter = Convolution->OutputsPerFilterCount;
    auto effectiveComputeHidden =  &CnnLayer::computeHidden;
    auto effectiveCompute = &CnnLayer::compute;
    if (Pooling)
    {
        outputsPerFilter = Pooling->OutputsPerFilterCount;
        // Activation is required for cnn with pooling
        ModelErrorHelper::ExpectNotNull(Activation.get(), Gna2ItemTypeOperationOperands, PwlOperandIndex);
        effectiveComputeHidden = &CnnLayer::computeHiddenPool;
        effectiveCompute = &CnnLayer::computePool;
    }
    else if (Activation)
    {
        effectiveComputeHidden = &CnnLayer::computeHiddenPwl;
        effectiveCompute = &CnnLayer::computePwl;
    }
    const auto& declaredOutputPerFilter = Output.AsModelValue('W').SetOperand(OutputOperandIndex);
    ModelErrorHelper::ExpectEqual(declaredOutputPerFilter, outputsPerFilter);

    Layer::ComputeHidden = [this, effectiveComputeHidden](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {(this->*effectiveComputeHidden)(accel, executionConfig); };

    Layer::Compute = [this, effectiveCompute](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
    {(this->*effectiveCompute)(layerConfiguration, accel, executionConfig); };
}

Tensor const & CnnLayer::GetOperand(uint32_t operandIndex) const
{
    // TODO:3:replace with generic solution when all layers are transforms
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        if (Activation)
        {
            return Output.ScratchPad;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
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

void CnnLayer::computePool(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ layerConfiguration.Configs.Convolution.get(), execution };
    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}

const Gna2Operation & CnnLayer::getDetails(const Gna2Operation & operation)
{
    return operation;
}
