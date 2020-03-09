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

#include "TransposeLayer.h"

#include "AccelerationDetector.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "Macros.h"

using namespace GNA;

TransposeLayer::TransposeLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    transposeKernels{ AccelerationDetector::GetKernelMap<TransposeKernel>(
                                            KERNEL_TRANSPOSE,  KernelMode{Input.Mode}) },
    transposeHiddenConfig(std::make_unique<TransposeConfig>(
                Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, Output.Buffer))
{
    Expect::Equal(Input.Dimensions.at('W'), Output.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);
    Expect::Equal(Input.Dimensions.at('H'), Output.Dimensions.at('W'), Gna2StatusXnnErrorLyrCfg);
    Expect::Null(layer.pLayerStruct); // transpose layers do not have layer details

    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

TransposeLayer::TransposeLayer(
        const Gna2Operation& apiOperation,
        const BaseValidator& validatorIn) :
    Layer(apiOperation, validatorIn, {}, BaseAddress{}),
    transposeKernels{ AccelerationDetector::GetKernelMap<TransposeKernel>(
                                            KERNEL_TRANSPOSE, KernelMode{Input.Mode}) }
{
    auto *inputTensor = reinterpret_cast<const Gna2Tensor *>(apiOperation.Operands[InputOperandIndex]);
    Expect::Equal(inputTensor->Shape.NumberOfDimensions,
            static_cast<uint32_t>(2), Gna2StatusXnnErrorLyrCfg);

    auto *outputTensor = reinterpret_cast<const Gna2Tensor *>(apiOperation.Operands[OutputOperandIndex]);
    Expect::Equal(outputTensor->Shape.NumberOfDimensions,
            static_cast<uint32_t>(2), Gna2StatusXnnErrorLyrCfg);

    transposeHiddenConfig = std::make_unique<TransposeConfig>(
            Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, Output.Buffer);

    ModelErrorHelper::ExpectEqual(Output.AsModelValue('H'), Input.AsModelValue('W'));
    ModelErrorHelper::ExpectEqual(Output.AsModelValue('W'), Input.AsModelValue('H'));

    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

void TransposeLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputOperandIndex) != 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputOperandIndex];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputOperandIndex) != 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputOperandIndex];
        Output.ValidateBuffer(outputBuffer);
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Transpose)
    {
        configs.Transpose = std::make_unique<TransposeConfig>(*transposeHiddenConfig);
    }

    configs.Transpose->input = inputBuffer;
    configs.Transpose->output = outputBuffer;
}

DataConfig TransposeLayer::GetDataMode() const
{
    return DataConfig(Input.Mode, GNA_DATA_DISABLED, GNA_DATA_DISABLED, Output.Mode);
}

void TransposeLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    transposeKernels.at(accel)(transposeHiddenConfig.get());
}

void TransposeLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    auto transposeConfig = layerConfiguration.Configs.Transpose.get();
    transposeKernels.at(accel)(transposeConfig);
}
