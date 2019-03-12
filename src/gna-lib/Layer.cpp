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

#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "LayerConfiguration.h"
#include "RecurrentLayer.h"
#include "SimpleLayers.h"
#include "Expect.h"

using std::make_unique;
using std::unique_ptr;

using namespace GNA;

unique_ptr<Layer> Layer::Create(const nn_layer* layer, const BaseValidator& validatorIn)
{
    switch (layer->operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
        return make_unique<AffineLayer>(layer, validatorIn);
    case INTEL_AFFINE_DIAGONAL:
        return make_unique<AffineDiagonalLayer>(layer, validatorIn);
    case INTEL_CONVOLUTIONAL:
        return make_unique<CnnLayer>(layer, validatorIn);
    case INTEL_CONVOLUTIONAL_2D:
        return make_unique<ConvolutionalLayer2D>(layer, validatorIn);
    case INTEL_COPY:
        return make_unique<CopyLayer>(layer, validatorIn);
    case INTEL_INTERLEAVE:/* FALLTHRU */
    case INTEL_DEINTERLEAVE:
        return make_unique<TransposeLayer>(layer, validatorIn);
    case INTEL_GMM:
        return make_unique<GmmLayer>(layer, validatorIn);
    case INTEL_RECURRENT:
        return make_unique<RnnLayer>(layer, validatorIn);
    default:
        return nullptr;
    }
}

Layer::Layer(const nn_layer *layer, const BaseValidator& validatorIn,
    const std::vector<TransformOperation>& transforms,
    const BaseAddress& intermediateBuffer) :
    validator{ make_unique<const LayerValidator>(validatorIn, layer->operation) },
    Operation{ layer->operation },
    Input{ *layer, *validator },
    Output{ *layer, *validator }
{
    Expect::InRange<uint32_t>(Operation, 0, LAYER_OPERATION_TYPE_COUT-1, XNN_ERR_LYR_OPERATION);

    //TODO:3: uncomment when all layers are Transform-based, remove if below
    //Expect::False(transforms.empty(), status_t::GNA_NULLARGNOTALLOWED); // TODO:3: add error code
    if (false == transforms.empty())
    {
        auto&& commonConfig = TransformFactoryConfig(&Input, &Output, Output.Mode, intermediateBuffer,
            layer->pLayerStruct, *validator);
        for (const auto& transform : transforms)
        {
            outputTransform = Transforms.Emplace(transform, commonConfig);
            commonConfig.input = outputTransform->Output.get();
        }

        inputTransform = Transforms.begin()->get();
        if (Output.Buffer)
            outputTransform->SetOutput(Output.Buffer);
    }
}

void Layer::addBufferAs(const BufferMap& source, GnaComponentType sourceType,
    BufferMap& destination, GnaComponentType destinationType) const
{
    if (IntermediateOutputComponent == sourceType && Transforms.size() < 2)
        return;

    auto buffer = source.find(sourceType);
    if (buffer != source.end())
    {
        destination[destinationType] = buffer->second;
    }
}

void Layer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    //TODO:3: remove condition when all layers use Transforms
    if (Transforms.size() > 0)
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

