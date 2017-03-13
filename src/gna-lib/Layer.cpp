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

#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "GmmLayer.h"
#include "Layer.h"
#include "RecurrentLayer.h"
#include "SimpleLayers.h"
#include "Validator.h"

using namespace GNA;
using std::make_unique;



const std::map<const nn_layer_kind, const Orientations> LayerConfig::OrientationsMap =
{
    { INTEL_AFFINE, INTERLEAVED },
    { INTEL_AFFINE_DIAGONAL, INTERLEAVED },
    { INTEL_AFFINE_MULTIBIAS, INTERLEAVED },
    { INTEL_CONVOLUTIONAL, FLAT },
    { INTEL_COPY, FLAT },
    { INTEL_DEINTERLEAVE, INTERLEAVED },
    { INTEL_GMM, INTERLEAVED },
    { INTEL_INTERLEAVE, FLAT },
    { INTEL_RECURRENT, FLAT }
};

LayerConfig::LayerConfig(const nn_layer_kind kind, const nn_layer_type type) :
    Kind(kind),
    Type(type),
    Orientation(OrientationsMap.at(kind))
{
    Validate::IsInRange(kind, 0, NUM_LAYER_KINDS, XNN_ERR_LYR_KIND);
    Validate::IsInRange(type, 0, NUM_LAYER_TYPES, XNN_ERR_LYR_TYPE);
};


LayerMatrix::LayerMatrix(const uint32_t rowCount, const uint32_t columnCount, void const * buffer, const LayerConfig& config) :
    ColumnCount(columnCount),
    RowCount(rowCount),
    ElementCount((FLAT == config.Orientation) ? ColumnCount : RowCount),
    Buffer(static_cast<void const * const>(buffer))
{
    Validate::IsInRange(config.Orientation, INTERLEAVED, FLAT, XNN_ERR_LYR_CFG);
    if (INTEL_HIDDEN == config.Type)
    {
        Validate::IsNull(Buffer);
        Validate::IsAlignedTo64(Buffer);
    }
};

LayerInput::LayerInput(const nn_layer &layer, const LayerConfig& config, const uint32_t vectorCount) :
    LayerMatrix(layer.nInputRows, layer.nInputColumns, layer.pInputs, config),
    VectorCount(vectorCount)
{
    if (INTEL_GMM == layer.nLayerKind)
    {
        Validate::IsTrue(layer.nBytesPerInput != GMM_FV_ELEMENT_SIZE, XNN_ERR_INPUT_BYTES);
    }
    else
    {
        Validate::IsTrue(layer.nBytesPerInput != 2, XNN_ERR_INPUT_BYTES);
    }
    Validate::IsInRange(VectorCount, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
    Validate::IsInRange(ElementCount, XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    Validate::IsMultiplicityOf(ElementCount, XNN_N_IN_ELEMS_MPLY);
    auto secondDimension = (FLAT == config.Orientation) ? RowCount : ColumnCount;
    Validate::IsInRange(secondDimension, 1, XNN_N_GROUP_MAX, XNN_ERR_GROUPING);
};


LayerOutput::LayerOutput(const nn_layer &layer, const LayerConfig& config) :
    LayerMatrix(layer.nOutputRows, layer.nOutputColumns, layer.pOutputs, config),
    ScratchPad(static_cast<uint32_t const * const>(layer.pOutputsIntermediate))
{
    Validate::IsInRange(ElementCount, 1, XNN_N_IN_ELEMS_MAX, XNN_ERR_LYR_CFG);
    Validate::IsInRange(layer.nBytesPerOutput, ActivatedOutputSize, NonActivatedOutputSize, XNN_ERR_INPUT_BYTES);
    Validate::IsTrue(NonActivatedOutputSize != layer.nBytesPerIntermediateOutput, XNN_ERR_INT_OUTPUT_BYTES);
    //Validate::IsNull(ScratchPad); // TODO: review when scratch-pad is allocated by gna-lib
    Validate::IsAlignedTo64(ScratchPad);
};

void LayerOutput::Validate(const bool ActivationEnabled, const uint32_t outputSize) const
{
    if (ActivationEnabled)
    {
        // if pwl is used 2B output buffer must be set
        Validate::IsNull(Buffer);
        Validate::IsTrue(ActivatedOutputSize != outputSize, XNN_ERR_INT_OUTPUT_BYTES);
    }
    else
    {
        Validate::IsTrue(NonActivatedOutputSize != outputSize, XNN_ERR_INT_OUTPUT_BYTES);
    }
}

unique_ptr<Layer> Layer::Create(const nn_layer* layer, const uint32_t inputVectorCount)
{
    switch (layer->nLayerKind)
    {
    case INTEL_AFFINE:
        return make_unique<AffineLayer>(layer, inputVectorCount);
    case INTEL_AFFINE_DIAGONAL:
        return make_unique<AffineDiagonalLayer>(layer, inputVectorCount);
    case INTEL_AFFINE_MULTIBIAS:
        return make_unique<AffineMultiBiasLayer>(layer, inputVectorCount);
    //case INTEL_CONVOLUTIONAL:
        //return new CnnLayer();
    case INTEL_COPY:
        return make_unique<CopyLayer>(layer);
    case INTEL_DEINTERLEAVE:
        return make_unique<TransposeLayer>(layer, inputVectorCount);
    case INTEL_GMM:
        return make_unique<GmmLayer>(layer, inputVectorCount);
    case INTEL_INTERLEAVE:
        return make_unique<TransposeLayer>(layer, inputVectorCount);
    //case INTEL_RECURRENT:
        //return new RnnLayer();
    default:
        return nullptr;
    }
}

Layer::Layer(const nn_layer *layer, const uint32_t inputVectorCount) :
    Config(layer->nLayerKind, layer->type),
    sourceLayer(validate(layer)),
    Input(sourceLayer, Config, inputVectorCount),
    Output(sourceLayer, Config)
{
}

const nn_layer Layer::validate(const nn_layer *layer)
{
    Validate::IsNull(layer);
    if (INTEL_INTERLEAVE != layer->nLayerKind && INTEL_DEINTERLEAVE != layer->nLayerKind)
    {
        Validate::IsNull(layer->pLayerStruct);
    }
    return *layer;
}