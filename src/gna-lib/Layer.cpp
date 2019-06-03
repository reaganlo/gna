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
        return std::make_unique<RnnLayer>(layer, validatorIn);
    default:
        return nullptr;
    }
}

std::unique_ptr<GNA::Layer> Layer::Create(const Gna2Operation & operation, const BaseValidator & validatorIn)
{
    ModelWrapper::ExpectOperationValid(operation);
    switch (operation.Type)
    {
    case Gna2OperationTypeFullyConnectedAffine:
    {
        if (operation.Operands[5] != nullptr)
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
        return std::make_unique<RnnLayer>(operation, validatorIn);
    case Gna2OperationTypeCopy:
        return std::make_unique<CopyLayer>(operation, validatorIn);
    case Gna2OperationTypeConvolution:
        return std::make_unique<ConvolutionalLayer2D>(operation, validatorIn);
    case Gna2OperationTypeTransposition:
        return std::make_unique<TransposeLayer>(operation, validatorIn);
    default:
        //TODO:3:P1:Add implementation for remaining operation types
        throw GnaException(Gna2StatusNotImplemented);
    }
}

void Layer::addBufferAs(const BufferMap& source, GnaComponentType sourceType,
    BufferMap& destination, GnaComponentType destinationType) const
{
    if (IntermediateOutputComponent == sourceType && Transforms.size() < 2)
    {
        return;
    }

    auto buffer = source.find(sourceType);
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
        nonIoBuffers.erase(InputComponent);
        nonIoBuffers.erase(OutputComponent);
        nonIoBuffers.erase(IntermediateOutputComponent);

        for (auto transform = Transforms.cbegin(); transform != Transforms.cend(); transform++)
        {
            BufferMap buffers = nonIoBuffers;
            if (transform == Transforms.cbegin())
            {
                addBufferAs(layerConfiguration.Buffers, InputComponent,
                    buffers, InputComponent);
                addBufferAs(layerConfiguration.Buffers, IntermediateOutputComponent,
                    buffers, OutputComponent);
            }
            if (transform == --Transforms.cend())
            {
                addBufferAs(layerConfiguration.Buffers, OutputComponent,
                    buffers, OutputComponent);
                addBufferAs(layerConfiguration.Buffers, IntermediateOutputComponent,
                    buffers, InputComponent);
            }
            if (transform != Transforms.cbegin() && transform != --Transforms.cend())
            {
                addBufferAs(layerConfiguration.Buffers, IntermediateOutputComponent,
                    buffers, InputComponent);
                addBufferAs(layerConfiguration.Buffers, IntermediateOutputComponent,
                    buffers, OutputComponent);
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

// TODO:3 support all component types
uint32_t Layer::GetOperandSize(GnaComponentType componentType) const
{
    switch (componentType)
    {
    case InputComponent:
        return Input.Size;
    case OutputComponent:
        return Output.Size;
    case IntermediateOutputComponent:
        return Output.ScratchPad.Size;
    default:
        return 0;
    }
}

void Layer::InitTransforms(const std::vector<TransformOperation>& transforms,
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
}

nn_operation AbstractOperation::toLegacy(
    const Gna2Operation& operation, const BaseValidator& validator)
{
    //TODO:3:P1: Add remaining cases
    switch(operation.Type)
    {
        case Gna2OperationTypeElementWiseAffine:
            return INTEL_AFFINE_DIAGONAL;
        case Gna2OperationTypeFullyConnectedAffine:
            if (AffineFunction::HasGroupedBias(operation))
            {
                return INTEL_AFFINE_MULTIBIAS;
            }
            return INTEL_AFFINE;
        case Gna2OperationTypeCopy:
            return INTEL_COPY;
        case Gna2OperationTypeTransposition:
        {
            const Gna2Tensor& inputTensor = *operation.Operands[0];
            if (LayerInput::IsTensorValid(inputTensor, validator, INTEL_INTERLEAVE))
            {
                return INTEL_INTERLEAVE;
            }
            if (LayerInput::IsTensorValid(inputTensor, validator, INTEL_DEINTERLEAVE))
            {
                return INTEL_DEINTERLEAVE;
            }
            throw GnaException { Gna2StatusXnnErrorLyrOperation };
        }
        case Gna2OperationTypeRecurrent:
            return INTEL_RECURRENT;
        default:
            if (operation.Type == Gna2OperationTypeConvolution &&
                operation.Operands[0]->Shape.NumberOfDimensions == 4)
            {
                return INTEL_CONVOLUTIONAL_2D;
            }
            throw GnaException(Gna2StatusNotImplemented);
    }
}

