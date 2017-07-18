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

#include "Layer.h"

#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "LayerConfiguration.h"
#include "RecurrentLayer.h"
#include "SimpleLayers.h"
#include "Validator.h"

using std::make_unique;
using std::unique_ptr;

using namespace GNA;

// FLAT -> ElementCount = columnCount
// FLAT -> VectorCount = rowCount
// INTERLEAVED -> ElementCount = rowCount
// INTERLEAVED -> VectorCount = columnCount 
const std::map<const nn_layer_kind, const Orientations> LayerConfig::InputOrientationsMap =
{
    { INTEL_AFFINE, INTERLEAVED },
    { INTEL_AFFINE_DIAGONAL, INTERLEAVED },
    { INTEL_AFFINE_MULTIBIAS, INTERLEAVED },
    { INTEL_CONVOLUTIONAL, FLAT },
    { INTEL_COPY, FLAT },
    { INTEL_DEINTERLEAVE, INTERLEAVED },
    { INTEL_GMM, FLAT },
    { INTEL_INTERLEAVE, FLAT },
    { INTEL_RECURRENT, FLAT }
};

const std::map<const nn_layer_kind, const Orientations> LayerConfig::OutputOrientationsMap =
{
    { INTEL_AFFINE, INTERLEAVED },
    { INTEL_AFFINE_DIAGONAL, INTERLEAVED },
    { INTEL_AFFINE_MULTIBIAS, INTERLEAVED },
    { INTEL_CONVOLUTIONAL, FLAT },
    { INTEL_COPY, FLAT },
    { INTEL_DEINTERLEAVE, FLAT },
    { INTEL_GMM, FLAT },
    { INTEL_INTERLEAVE, INTERLEAVED },
    { INTEL_RECURRENT, FLAT }
};

LayerConfig::LayerConfig(const nn_layer_kind kind, const nn_layer_type type) :
    Kind{kind},
    Type{type},
    InputOrientation{InputOrientationsMap.at(kind)},
    OutputOrientation{OutputOrientationsMap.at(kind)}
{
    Expect::InRange(kind, 0, NUM_LAYER_KINDS-1, XNN_ERR_LYR_TYPE);
    Expect::InRange(type, 0, NUM_LAYER_TYPES-1, XNN_ERR_LYR_TYPE);
};

LayerMatrix::LayerMatrix(const uint32_t rowCount, const uint32_t columnCount, const uint32_t elementSize, void const * buffer,
    const Orientations orientation, const nn_layer_type layerType) :
    ElementCount{(FLAT == orientation) ? columnCount : rowCount},
    VectorCount{(FLAT == orientation) ? rowCount : columnCount},
    Buffer{buffer},
    BufferSize{rowCount * columnCount * elementSize}
{
    Expect::InRange(orientation, INTERLEAVED, FLAT, XNN_ERR_LYR_CFG);
    Expect::InRange(VectorCount, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    if (INTEL_HIDDEN == layerType)
    {
        Expect::ValidBuffer(Buffer);
    }
};

LayerInput::LayerInput(const nn_layer &layer, const Orientations orientation, const nn_layer_type layerType) :
    LayerMatrix{layer.nInputRows, layer.nInputColumns, layer.nBytesPerInput, layer.pInputs, orientation, layerType}
{
    if (INTEL_GMM == layer.nLayerKind)
    {
        Expect::True(layer.nBytesPerInput == GMM_FV_ELEMENT_SIZE, XNN_ERR_INPUT_BYTES);
    }
    else
    {
        Expect::True(layer.nBytesPerInput == 2, XNN_ERR_INPUT_BYTES);
    }
    Expect::InRange(ElementCount, XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(ElementCount, XNN_N_IN_ELEMS_MPLY);
    if (INTEL_OUTPUT == layerType)
    {
        Expect::ValidBuffer(Buffer);
    }
};

LayerOutput::LayerOutput(const nn_layer& layer, const Orientations orientation, const nn_layer_type layerType) :
    LayerMatrix{layer.nOutputRows, layer.nOutputColumns, layer.nBytesPerOutput, layer.pOutputs, orientation, layerType},
    ScratchPad{static_cast<int32_t * const>(layer.pOutputsIntermediate)},
    mode{NonActivatedOutput}
{
    if (INTEL_GMM == layer.nLayerKind)
    {
        Expect::InRange(ElementCount, 1, GMM_STATES_COUNT_MAX, XNN_ERR_LYR_CFG);
    }
    else
    {
        Expect::InRange(ElementCount, 1, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    }
    Expect::InRange(layer.nBytesPerOutput, ActivatedOutputSize, NonActivatedOutputSize, XNN_ERR_INPUT_BYTES);
    Expect::True(NonActivatedOutputSize == layer.nBytesPerIntermediateOutput, XNN_ERR_INT_OUTPUT_BYTES);
    //Expect::ValidBuffer(ScratchPad); // TODO: review when scratch-pad is allocated by gna-lib
    if (INTEL_INPUT == layerType)
    {
        Expect::ValidBuffer(Buffer);
    }
};

void LayerOutput::SetOutputMode(const bool activationEnabled, const uint32_t outputSize)
{
    if (activationEnabled)
    {
        Expect::ValidBuffer(ScratchPad); // intermediate output buffer must be set always
        Expect::True(ActivatedOutputSize == outputSize, XNN_ERR_INT_OUTPUT_BYTES);
        mode = ActivatedOutput;
    }
    else
    {
        Expect::True(NonActivatedOutputSize == outputSize, XNN_ERR_INT_OUTPUT_BYTES);
        mode = NonActivatedOutput;
    }
}

unique_ptr<Layer> Layer::Create(const nn_layer* layer)
{
    switch (layer->nLayerKind)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
        return make_unique<AffineLayer>(layer);
    case INTEL_AFFINE_DIAGONAL:
        return make_unique<AffineDiagonalLayer>(layer);
    case INTEL_CONVOLUTIONAL:
        return make_unique<CnnLayer>(layer);
    case INTEL_COPY:
        return make_unique<CopyLayer>(layer);
    case INTEL_INTERLEAVE:
        /* FALLTHRU */
    case INTEL_DEINTERLEAVE:
        return make_unique<TransposeLayer>(layer);
    case INTEL_GMM:
        return make_unique<GmmLayer>(layer);
    case INTEL_RECURRENT:
        return make_unique<RnnLayer>(layer);
    default:
        return nullptr;
    }
}

Layer::Layer(const nn_layer *layer) :
    Config{layer->nLayerKind, layer->type},
    Input{*layer, Config.InputOrientation, Config.Type},
    Output{*layer, Config.OutputOrientation, Config.Type}
{
}

void Layer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration, ValidBoundariesFunctor validBoundaries) const
{
    if (layerConfiguration.ActiveList)
    {
        const auto activeList = layerConfiguration.ActiveList.get();
        validBoundaries(activeList->Indices, activeList->IndicesCount * sizeof(uint32_t));
    }
}
