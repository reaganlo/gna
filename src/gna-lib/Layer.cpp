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

#include "Layer.h"

#include "AffineFunctions.h"
#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalLayer2D.h"
#include "CopyLayer.h"
#include "DataMode.h"
#include "Expect.h"
#include "GmmLayer.h"
#include "LayerConfiguration.h"
#include "ModelWrapper.h"
#include "RecurrentLayer.h"
#include "TransposeLayer.h"

#include <map>
#include <utility>

using namespace GNA;

std::unique_ptr<Layer> Layer::Create(const nn_layer& layer, const BaseValidator& validatorIn)
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
        return std::make_unique<AffineLayer>(layer, validatorIn);
    case INTEL_AFFINE_DIAGONAL:
        return std::make_unique<AffineDiagonalLayer>(layer, validatorIn);
    case INTEL_CONVOLUTIONAL:
        return std::make_unique<CnnLayer>(layer, validatorIn);
    case INTEL_CONVOLUTIONAL_2D:
        return std::make_unique<ConvolutionalLayer2D>(layer, validatorIn);
    case INTEL_COPY:
        return std::make_unique<CopyLayer>(layer, validatorIn);
    case INTEL_INTERLEAVE:/* FALLTHRU */
    case INTEL_DEINTERLEAVE:
        return std::make_unique<TransposeLayer>(layer, validatorIn);
    case INTEL_GMM:
        return std::make_unique<GmmLayer>(layer, validatorIn);
    case INTEL_RECURRENT:
        return std::make_unique<RecurrentLayer>(layer, validatorIn);
    default:
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}

std::unique_ptr<GNA::Layer> Layer::Create(const Gna2Operation & operation, const BaseValidator & validatorIn)
{
    ModelWrapper::ExpectOperationValid(operation);
    switch (operation.Type)
    {
    case Gna2OperationTypeFullyConnectedAffine:
    {
        if (operation.Operands[WeightScaleFactorOperandIndex] != nullptr)
        {
            Expect::NotNull(operation.Parameters);
            Expect::NotNull(operation.Parameters[0]);
            Expect::NotNull(operation.Parameters[1]);
        }
        return std::make_unique<AffineLayer>(operation, validatorIn);
    }
    case Gna2OperationTypeElementWiseAffine:
        return std::make_unique<AffineLayer>(operation, validatorIn);
    case Gna2OperationTypeRecurrent:
        return std::make_unique<RecurrentLayer>(operation, validatorIn);
    case Gna2OperationTypeCopy:
        return std::make_unique<CopyLayer>(operation, validatorIn);
    case Gna2OperationTypeConvolution:
        //TODO:3:Replace with elegant dispatcher or cover every case by ConvolutionalLayer2D
        if (ConvolutionalLayer2D::IsSupported(operation))
        {
            return std::make_unique<ConvolutionalLayer2D>(operation, validatorIn);
        }
        else
        {
            return std::make_unique<CnnLayer>(operation, validatorIn);
        }
        /*case Gna2OperationTypeGmm:
            return std::make_unique<GmmLayer>(operation, validatorIn);*/
    case Gna2OperationTypeTransposition:
        return std::make_unique<TransposeLayer>(operation, validatorIn);
    default:
        //TODO:3:P1:Add implementation for remaining operation types
        throw GnaException(Gna2StatusNotImplemented);
    }
}

void Layer::addBufferAs(const BufferMap& source, uint32_t sourceType,
    BufferMap& destination, uint32_t destinationType) const
{
    if (ScratchpadOperandIndex == sourceType && Transforms.size() < 2)
    {
        return;
    }

    const auto buffer = source.find(sourceType);
    if (buffer != source.end())
    {
        destination[destinationType] = buffer->second;
    }
}

void Layer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    //TODO:3: remove condition when all layers use Transforms
    if (!Transforms.empty())
    {
        auto nonIoBuffers = layerConfiguration.Buffers;
        nonIoBuffers.erase(InputOperandIndex);
        nonIoBuffers.erase(OutputOperandIndex);
        nonIoBuffers.erase(ScratchpadOperandIndex);

        for (auto transform = Transforms.cbegin(); transform != Transforms.cend(); ++transform)
        {
            BufferMap buffers = nonIoBuffers;
            if (transform == Transforms.cbegin())
            {
                addBufferAs(layerConfiguration.Buffers, InputOperandIndex,
                    buffers, InputOperandIndex);
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, OutputOperandIndex);
            }
            if (transform == --Transforms.cend())
            {
                addBufferAs(layerConfiguration.Buffers, OutputOperandIndex,
                    buffers, OutputOperandIndex);
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, InputOperandIndex);
            }
            if (transform != Transforms.cbegin() && transform != --Transforms.cend())
            {
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, InputOperandIndex);
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, OutputOperandIndex);
            }
            transform->get()->UpdateConfigBuffers(layerConfiguration.ConfigList, buffers);
        }
    }
}

// TODO:3 add methods to all layers
DataConfig Layer::GetDataMode() const
{
    return DataConfig(Input.Mode, GNA_INT16, GNA_INT32, Output.Mode);
}

Tensor const & Layer::GetOperand(uint32_t operandIndex) const
{

    switch (operandIndex)
    {
    case InputOperandIndex:
        return Input;
    case OutputOperandIndex:
        return Output;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}

Tensor const * Layer::TryGetOperand(uint32_t operandIndex) const
{
    try
    {
        return &GetOperand(operandIndex);
    }
    catch (const GnaException&)
    {
        return nullptr;
    }
}

uint32_t Layer::TryGetOperandSize(uint32_t operandIndex) const
{
    auto const operand = TryGetOperand(operandIndex);
    if (nullptr != operand)
    {
        return operand->Size;
    }
    return 0;
}

void Layer::VerifyHas1BInputAnd2BWeight()
{
    if (is1BInputAnd2BWeightVerified)
    {
        return;
    }

    is1BInputAnd2BWeightVerified = true;

    auto const input = TryGetOperand(InputOperandIndex);
    auto const weight = TryGetOperand(WeightOperandIndex);
    if (input &&
        weight &&
        Gna2DataTypeInt8 == input->Mode &&
        Gna2DataTypeInt16 == weight->Mode)
    {
        has1BInputAnd2BWeight = true;
    }
}

Tensor const & Layer::getTransformOperand(TransformOperation operation, uint32_t operandIndex) const
{
    auto const transform = Transforms.Get(operation);
    if (transform)
    {
        return transform->GetOperand(operandIndex);
    }
    else
    {
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}

void Layer::initTransforms(const std::vector<TransformOperation>& transforms,
    TransformFactoryConfig & commonConfig, const OperationConfig & operationConfig)
{
    for (const auto& transform : transforms)
    {
        outputTransform = Transforms.Emplace(transform, commonConfig, operationConfig);
        commonConfig.input = outputTransform->Output.get();
    }

    inputTransform = Transforms.begin()->get();
    if (Output.Buffer)
    {
        outputTransform->SetOutput(Output.Buffer);
    }

    if (transforms.back() == ActivationTransform
        && outputTransform->Operation != ActivationTransform)
    {
        Expect::Equal(outputTransform->Output->Mode.Type, Gna2DataTypeInt32,
            Gna2StatusXnnErrorOutputBytes);
    }
}

void Layer::initComputeFunctions()
{
    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(nullptr, accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration,
        AccelerationMode accel,
        ExecutionConfig const & executionConfig)
    {this->compute(&layerConfiguration, accel, executionConfig); };
}

nn_operation AbstractOperation::toLegacy(
    const Gna2Operation& operation, const BaseValidator& validator)
{
    //TODO:3:P1: Add remaining cases
    switch (operation.Type)
    {
    case Gna2OperationTypeElementWiseAffine:
        return INTEL_AFFINE_DIAGONAL;
    case Gna2OperationTypeFullyConnectedAffine:
        if (OperationConfig::IsMultibias(operation))
        {
            return INTEL_AFFINE_MULTIBIAS;
        }
        return INTEL_AFFINE;
    case Gna2OperationTypeCopy:
        return INTEL_COPY;
    case Gna2OperationTypeTransposition:
    {
        const Gna2Tensor& inputTensor = *operation.Operands[InputOperandIndex];
        if (LayerInput::IsTensorValid(inputTensor, validator, INTEL_INTERLEAVE))
        {
            return INTEL_INTERLEAVE;
        }
        if (LayerInput::IsTensorValid(inputTensor, validator, INTEL_DEINTERLEAVE))
        {
            return INTEL_DEINTERLEAVE;
        }
        throw GnaException{ Gna2StatusXnnErrorLyrOperation };
    }
    case Gna2OperationTypeRecurrent:
        return INTEL_RECURRENT;
    case Gna2OperationTypeConvolution:
        if (ConvolutionalLayer2D::IsSupported(operation))
        {
            return INTEL_CONVOLUTIONAL_2D;
        }
        return INTEL_CONVOLUTIONAL;
    case Gna2OperationTypeGmm:
        return INTEL_GMM;
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}

Gna2OperationType AbstractOperation::fromLegacy(const nn_operation& layerType)
{
    static const std::map<nn_operation, Gna2OperationType> operationTypes =
    {
        {INTEL_AFFINE, Gna2OperationTypeFullyConnectedAffine},
        {INTEL_AFFINE_DIAGONAL, Gna2OperationTypeElementWiseAffine},
        {INTEL_AFFINE_MULTIBIAS, Gna2OperationTypeFullyConnectedAffine},
        {INTEL_CONVOLUTIONAL, Gna2OperationTypeConvolution},
        {INTEL_CONVOLUTIONAL_2D, Gna2OperationTypeConvolution},
        {INTEL_COPY, Gna2OperationTypeCopy},
        {INTEL_DEINTERLEAVE, Gna2OperationTypeTransposition},
        {INTEL_GMM, Gna2OperationTypeGmm},
        {INTEL_INTERLEAVE, Gna2OperationTypeTransposition},
        {INTEL_RECURRENT, Gna2OperationTypeRecurrent},
    };

    try
    {
        return operationTypes.at(layerType);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

