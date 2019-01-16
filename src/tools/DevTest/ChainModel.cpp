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

#include "ChainModel.h"
#include "ModelUtilities.h"

#define UNREFERENCED_PARAMETER(P) ((void)(P))

ChainModel::ChainModel()
{
    nnet.nGroup = 4;
    nnet.nLayers = 0;
    nnet.pLayers = nullptr;
}

ChainModel::~ChainModel()
{
    for (auto& layer : layers)
    {
        if (layer.pLayerStruct) free(layer.pLayerStruct);
    }
}

ChainModel& ChainModel::Affine(bool weights2B, bool pwlEnabled, bool activeListEnabled)
{
    // TODO: active list
    UNREFERENCED_PARAMETER(activeListEnabled);

    if (locked) throw;

    intel_affine_func_t affine_func;
    affine_func.nBytesPerWeight = weights2B ? GNA_INT16 : GNA_INT8;
    affine_func.nBytesPerBias = weights2B ? GNA_INT32: GNA_DATA_RICH_FORMAT;

    auto affine_layer = static_cast<intel_affine_layer_t*>(calloc(1, sizeof(intel_affine_layer_t)));
    affine_layer->affine = affine_func;
    if (pwlEnabled)
    {
        affine_layer->pwl.nSegments = 64;
    }
    else
    {
        affine_layer->pwl.nSegments = 0;
        affine_layer->pwl.pSegments = nullptr;
    }

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = groupingNum;
    nnet_layer.nInputRows = inVecSz;
    nnet_layer.nOutputColumns = groupingNum;
    nnet_layer.nOutputRows = outVecSz;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_AFFINE;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = affine_layer;

    if (pwlEnabled)
    {
        nnet_layer.nBytesPerOutput = GNA_INT16;
    }
    else
    {
        nnet_layer.nBytesPerOutput = GNA_INT32;
    }

    modelSize += ModelUtilities::CalculateDnnSize(groupingNum, inVecSz, outVecSz, weights2B ? sizeof(int16_t) : sizeof(int8_t),
        pwlEnabled ? nSegments : 0);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Diagonal(bool weights2B, bool pwlEnabled)
{
    if (locked) throw;

    intel_affine_func_t affine_func;
    affine_func.nBytesPerWeight = weights2B ? GNA_INT16 : GNA_INT8;
    affine_func.nBytesPerBias = weights2B ? GNA_INT32: GNA_DATA_RICH_FORMAT;

    auto affine_layer = static_cast<intel_affine_layer_t*>(calloc(1, sizeof(intel_affine_layer_t)));
    affine_layer->affine = affine_func;
    if (pwlEnabled)
    {
        affine_layer->pwl.nSegments = 64;
    }
    else
    {
        affine_layer->pwl.nSegments = 0;
        affine_layer->pwl.pSegments = nullptr;
    }

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = groupingNum;
    nnet_layer.nInputRows = inVecSz;
    nnet_layer.nOutputColumns = groupingNum;
    nnet_layer.nOutputRows = outVecSz;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_AFFINE_DIAGONAL;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = affine_layer;

    if (pwlEnabled)
    {
        nnet_layer.nBytesPerOutput = GNA_INT16;
    }
    else
    {
        nnet_layer.nBytesPerOutput = GNA_INT32;
    }

    modelSize += ModelUtilities::CalculateDnnSize(groupingNum, inVecSz, outVecSz, weights2B ? sizeof(int16_t) : sizeof(int8_t),
        pwlEnabled ? nSegments : 0);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Multibias(bool weights2B, bool pwlEnabled)
{
    if (locked) throw;

    intel_affine_multibias_func_t multibias_func;
    multibias_func.nBytesPerWeight = weights2B ? GNA_INT16 : GNA_INT8;
    multibias_func.biasVectorCount = 4;
    multibias_func.biasVectorIndex = 1;

    auto multibias_layer = static_cast<intel_affine_multibias_layer_t*>(calloc(1, sizeof(intel_affine_multibias_layer_t)));
    multibias_layer->affine = multibias_func;
    if (pwlEnabled)
    {
        multibias_layer->pwl.nSegments = 64;
    }
    else
    {
        multibias_layer->pwl.nSegments = 0;
        multibias_layer->pwl.pSegments = nullptr;
    }

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = nnet.nGroup;
    nnet_layer.nInputRows = inVecSz;
    nnet_layer.nOutputColumns = nnet.nGroup;
    nnet_layer.nOutputRows = outVecSz;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_AFFINE_MULTIBIAS;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = multibias_layer;

    if (pwlEnabled)
    {
        nnet_layer.nBytesPerOutput = GNA_INT16;
    }
    else
    {
        nnet_layer.nBytesPerOutput = GNA_INT32;
    }

    modelSize += ModelUtilities::CalculateMultibiasSize(groupingNum, inVecSz, outVecSz, weights2B ? sizeof(int16_t) : sizeof(int8_t),
        pwlEnabled ? nSegments : 0);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Convolution(bool pwlEnabled)
{
    if (locked) throw;

    auto convolution_layer = static_cast<intel_convolutional_layer_t*>(calloc(1, sizeof(intel_convolutional_layer_t)));
    convolution_layer->nBytesBias = sizeof(intel_bias_t);
    convolution_layer->nBytesFilterCoefficient = sizeof(int16_t);
    convolution_layer->nFeatureMaps = 1;
    convolution_layer->nFeatureMapRows = 1;
    convolution_layer->nFeatureMapColumns = 48;
    convolution_layer->nFilters = 4;
    convolution_layer->nFilterRows = 1;
    convolution_layer->nFilterCoefficients = 48;
    convolution_layer->poolType = INTEL_NO_POOLING;

    if (pwlEnabled)
    {
        convolution_layer->pwl.nSegments = 64;
    }
    else
    {
        convolution_layer->pwl.nSegments = 0;
        convolution_layer->pwl.pSegments = nullptr;
    }

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = cnnInVecSz;
    nnet_layer.nInputRows = 1;
    nnet_layer.nOutputColumns = outVecSz;
    nnet_layer.nOutputRows = 1;
    if (pwlEnabled)
    {
        nnet_layer.nBytesPerOutput = GNA_INT16; // activated
    }
    else
    {
        nnet_layer.nBytesPerOutput = GNA_INT32;
    }
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_CONVOLUTIONAL;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = convolution_layer;

    auto outputsPerFilter = (nnet_layer.nInputColumns - convolution_layer->nFilterCoefficients)
        / (convolution_layer->nFeatureMaps * convolution_layer->nFeatureMapColumns) + 1;
    modelSize += ModelUtilities::CalculateCnnSize(cnnInVecSz, outputsPerFilter, convolution_layer->nFilters, convolution_layer->nFilterCoefficients, 64);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Pooling(intel_pool_type_t poolingType)
{
    if (locked) throw;

    auto convolution_layer = static_cast<intel_convolutional_layer_t*>(calloc(1, sizeof(intel_convolutional_layer_t)));
    convolution_layer->nBytesBias = sizeof(intel_bias_t);
    convolution_layer->nBytesFilterCoefficient = sizeof(int16_t);
    convolution_layer->nFeatureMaps = 1;
    convolution_layer->nFeatureMapRows = 1;
    convolution_layer->nFeatureMapColumns = 48;
    convolution_layer->nFilters = 4;
    convolution_layer->nFilterRows = 1;
    convolution_layer->nFilterCoefficients = 48;
    convolution_layer->poolType = poolingType;
    convolution_layer->nPoolSize = 6;
    convolution_layer->nPoolStride = 6;
    convolution_layer->pwl.nSegments = 64;

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = cnnInVecSz;
    nnet_layer.nInputRows = 1;
    nnet_layer.nOutputColumns = cnnOutVecSz;
    nnet_layer.nOutputRows = 1;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT16; // activated
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_CONVOLUTIONAL;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = convolution_layer;


    auto maxNCOE = (nnet_layer.nInputColumns - convolution_layer->nFilterCoefficients)
        / (convolution_layer->nFeatureMaps * convolution_layer->nFeatureMapColumns) + 1;
    auto outputsPerFilter = (maxNCOE - 1) / convolution_layer->nPoolStride + 1;
    modelSize += ModelUtilities::CalculateCnnSize(inVecSz, outputsPerFilter, convolution_layer->nFilters, convolution_layer->nFilterCoefficients, 64);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Recurrent(bool weights2B)
{
    if (locked) throw;

    intel_affine_func_t affine_func;
    affine_func.nBytesPerWeight = weights2B ? GNA_INT16 : GNA_INT8;
    affine_func.nBytesPerBias = weights2B ? GNA_INT32: GNA_DATA_RICH_FORMAT;

    auto recurrent_layer = static_cast<intel_recurrent_layer_t*>(calloc(1, sizeof(intel_recurrent_layer_t)));
    recurrent_layer->affine = affine_func;
    recurrent_layer->feedbackFrameDelay = 3;
    recurrent_layer->pwl.nSegments = 64;

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = inVecSz;
    nnet_layer.nInputRows = nnet.nGroup;
    nnet_layer.nOutputColumns = rnnOutVecSz;
    nnet_layer.nOutputRows = nnet.nGroup;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_RECURRENT;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = recurrent_layer;

    modelSize += ModelUtilities::CalculateRnnSize(nnet.nGroup, inVecSz, outVecSz, affine_func.nBytesPerWeight, 64);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Gmm()
{
    if (locked) throw;

    auto gmm = static_cast<gna_gmm_layer*>(calloc(1, sizeof(gna_gmm_layer)));
    gmm->config.layout = GMM_LAYOUT_FLAT;
    gmm->config.maximumScore = UINT32_MAX;
    gmm->config.mixtureComponentCount = 1;
    gmm->config.mode = GNA_MAXMIX16;
    gmm->config.stateCount = 8;

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = gmmInVecSz;
    nnet_layer.nInputRows = nnet.nGroup;
    nnet_layer.nOutputColumns = outVecSz;
    nnet_layer.nOutputRows = nnet.nGroup;
    nnet_layer.nBytesPerInput = GNA_INT8;
    nnet_layer.nBytesPerOutput = GNA_INT32;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32; // this is always 4 bytes
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.operation = INTEL_GMM;
    nnet_layer.pLayerStruct = gmm;

    modelSize += ModelUtilities::CalculateGmmSize(gmm->config.mixtureComponentCount,
        nnet.nGroup, inVecSz, gmm->config.stateCount, gmm->config.mode);
    layers.push_back(nnet_layer);
    GmmCount++;
    return *this;
}

ChainModel& ChainModel::Copy()
{
    if (locked) throw;

    intel_copy_layer_t *copy_layer = (intel_copy_layer_t*)calloc(1, sizeof(intel_copy_layer_t));
    copy_layer->nCopyCols = inVecSz;
    copy_layer->nCopyRows = groupingNum;

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = inVecSz;
    nnet_layer.nInputRows = nnet.nGroup;
    nnet_layer.nOutputColumns = inVecSz;
    nnet_layer.nOutputRows = nnet.nGroup;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_COPY;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = copy_layer;

    modelSize += ModelUtilities::CalculateSimpleSize(nnet.nGroup, inVecSz, outVecSz);
    layers.push_back(nnet_layer);
    return *this;
}

ChainModel& ChainModel::Transpose()
{
    if (locked) throw;

    intel_nnet_layer_t nnet_layer;
    nnet_layer.nInputColumns = inVecSz;
    nnet_layer.nInputRows = nnet.nGroup;
    nnet_layer.nOutputColumns = nnet.nGroup;
    nnet_layer.nOutputRows = inVecSz;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT16;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.operation = INTEL_INTERLEAVE;
    nnet_layer.mode = INTEL_HIDDEN;
    nnet_layer.pLayerStruct = nullptr;

    modelSize += ModelUtilities::CalculateSimpleSize(nnet.nGroup, nnet_layer.nInputColumns, nnet_layer.nInputColumns);
    layers.push_back(nnet_layer);
    return *this;
}

intel_nnet_type_t& ChainModel::Setup(uint8_t *pinned_memory)
{
    if (locked) throw;

    nnet.nLayers = static_cast<uint32_t>(layers.size());
    nnet.pLayers = &layers[0];
    nnet.pLayers[0].mode = INTEL_INPUT;
    nnet.pLayers[nnet.nLayers - 1].mode = INTEL_OUTPUT;
    locked = true;

    for (auto layerIx = uint32_t{0}; layerIx < nnet.nLayers; layerIx++)
    {
        auto layer = nnet.pLayers + layerIx;
        switch (layer->operation)
        {
        case INTEL_AFFINE:
            /* FALLTHRU */
        case INTEL_AFFINE_DIAGONAL:
            setup_dnn_pointers(layer, pinned_memory);
            break;
        case INTEL_AFFINE_MULTIBIAS:
            setup_multibias_pointers(layer, pinned_memory);
            break;
        case INTEL_CONVOLUTIONAL:
            setup_cnn_pointers(layer, pinned_memory);
            break;
        case INTEL_RECURRENT:
            setup_rnn_pointers(layer, pinned_memory);
            break;
        case INTEL_INTERLEAVE:
            /* FALLTHRU */
        case INTEL_DEINTERLEAVE:
            /* FALLTHRU */
        case INTEL_COPY:
            setup_simple_pointers(layer, pinned_memory);
            break;
        case INTEL_GMM:
            setup_gmm_pointers(layer, pinned_memory);
            break;
        default:
            break;
        }
    }

    return nnet;
}

uint32_t ChainModel::GetModelSize()
{
    return static_cast<uint32_t>(modelSize);
}

uint16_t ChainModel::GetLayerCount() const
{
    return static_cast<uint16_t>(layers.size());
}

uint32_t ChainModel::GetInputBuffersSize()
{
    if (!locked) throw;
    auto firstLayer = nnet.pLayers;
    auto inputBufferSize = ALIGN64(firstLayer->nInputRows * firstLayer->nInputColumns * sizeof(int16_t));
    return inputBufferSize;
}

uint32_t ChainModel::GetOutputBuffersSize()
{
    if (!locked) throw;
    auto lastLayer = nnet.pLayers + nnet.nLayers - 1;
    auto outputBufferSize = std::uint32_t{ lastLayer->nOutputRows * lastLayer->nOutputColumns };
    switch (lastLayer->operation)
    {
        case INTEL_INTERLEAVE:
            /* FALLTHRU */
        case INTEL_DEINTERLEAVE:
            /* FALLTHRU */
        case INTEL_COPY:
            /* FALLTHRU */
        case INTEL_RECURRENT:
            outputBufferSize *= sizeof(int16_t);
            break;
        case INTEL_GMM:
            outputBufferSize *= sizeof(int32_t);
            break;
        case INTEL_AFFINE:
        case INTEL_AFFINE_DIAGONAL:
            {
                auto affine_layer = static_cast<intel_affine_layer_t*>(lastLayer->pLayerStruct);
                outputBufferSize *= (affine_layer->pwl.nSegments > 0) ? sizeof(int16_t) : sizeof(int32_t);
                break;
            }
        case INTEL_AFFINE_MULTIBIAS:
            {
                auto affine_layer = static_cast<intel_affine_multibias_layer_t*>(lastLayer->pLayerStruct);
                outputBufferSize *= (affine_layer->pwl.nSegments > 0) ? sizeof(int16_t) : sizeof(int32_t);
                break;
            }
        case INTEL_CONVOLUTIONAL:
            {
                auto convolution_layer = static_cast<intel_convolutional_layer_t*>(lastLayer->pLayerStruct);
                outputBufferSize *= (convolution_layer->pwl.nSegments > 0) ? sizeof(int16_t) : sizeof(int32_t);
                break;
            }
        default:
            break;
    }
    return outputBufferSize;
}

void ChainModel::setup_dnn_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory)
{
    auto affine_layer = static_cast<intel_affine_layer_t*>(layer->pLayerStruct);

    void *pinned_inputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_inputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * layer->nInputColumns * sizeof(int16_t));
    }

    void *pinned_scratchpad = nullptr;
    auto bytesPerOutput = sizeof(int32_t);
    if (affine_layer->pwl.nSegments > 0)
    {
        bytesPerOutput = sizeof(int16_t);

        pinned_scratchpad = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * sizeof(int32_t));

        affine_layer->pwl.pSegments = (intel_pwl_segment_t*)pinned_memory;
        ModelUtilities::GeneratePwlSegments(affine_layer->pwl.pSegments, affine_layer->pwl.nSegments);
        pinned_memory += ALIGN64(affine_layer->pwl.nSegments * sizeof(intel_pwl_segment_t));
    }

    void *pinned_outputs = nullptr;
    if (INTEL_OUTPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_outputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * bytesPerOutput);
    }

    void *pinned_biases = pinned_memory;
    pinned_memory += ALIGN64(layer->nOutputRows * affine_layer->affine.nBytesPerBias);

    void *pinned_weights = pinned_memory;
    pinned_memory += ALIGN64(layer->nInputRows * layer->nOutputRows * affine_layer->affine.nBytesPerWeight);

    layer->pInputs = pinned_inputs;
    layer->pOutputs = pinned_outputs;
    layer->pOutputsIntermediate = pinned_scratchpad;

    affine_layer->affine.pBiases = pinned_biases;
    affine_layer->affine.pWeights = pinned_weights;
}

void ChainModel::setup_multibias_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory)
{
    auto affine_layer = static_cast<intel_affine_multibias_layer_t*>(layer->pLayerStruct);

    void *pinned_inputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_inputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * layer->nInputColumns * sizeof(int16_t));
    }

    void *pinned_scratchpad = nullptr;
    auto bytesPerOutput = sizeof(int32_t);
    if (affine_layer->pwl.nSegments > 0)
    {
        bytesPerOutput = sizeof(int16_t);

        pinned_scratchpad = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * sizeof(int32_t));

        affine_layer->pwl.pSegments = (intel_pwl_segment_t*)pinned_memory;
        ModelUtilities::GeneratePwlSegments(affine_layer->pwl.pSegments, affine_layer->pwl.nSegments);
        pinned_memory += ALIGN64(affine_layer->pwl.nSegments * sizeof(intel_pwl_segment_t));
    }

    void *pinned_outputs = nullptr;
    if (INTEL_OUTPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_outputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * bytesPerOutput);
    }

    void *pinned_biases = pinned_memory;
    pinned_memory += ALIGN64(layer->nOutputRows * sizeof(intel_bias_t));

    void *pinned_weights = pinned_memory;
    pinned_memory += ALIGN64(layer->nInputRows * layer->nOutputRows * affine_layer->affine.nBytesPerWeight);

    void *pinned_scales = pinned_memory;
    if (sizeof(int8_t) == affine_layer->affine.nBytesPerWeight)
        pinned_memory += ALIGN64(layer->nOutputRows * sizeof(intel_compound_bias_t));

    layer->pInputs = pinned_inputs;
    layer->pOutputs = pinned_outputs;
    layer->pOutputsIntermediate = pinned_scratchpad;

    affine_layer->affine.pBiases = (intel_bias_t*)pinned_biases;
    affine_layer->affine.pWeights = pinned_weights;
    affine_layer->affine.weightScaleFactors = (intel_weight_scaling_factor_t*)pinned_scales;
}

void ChainModel::setup_cnn_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory)
{
    auto cnn_layer = static_cast<intel_convolutional_layer_t*>(layer->pLayerStruct);

    void *pinned_inputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_inputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * layer->nInputColumns * sizeof(int16_t));
    }

    uint32_t nOutputsPerFilter = (layer->nInputColumns - cnn_layer->nFilterCoefficients)
        / (cnn_layer->nFeatureMaps * cnn_layer->nFeatureMapColumns) + 1;
    if (INTEL_NO_POOLING != cnn_layer->poolType)
    {
        nOutputsPerFilter = (nOutputsPerFilter - 1) / cnn_layer->nPoolStride + 1;
    }

    void *pinned_scratchpad = nullptr;
    auto bytesPerOutput = sizeof(int32_t);
    if (cnn_layer->pwl.nSegments > 0)
    {
        bytesPerOutput = sizeof(int16_t);

        pinned_scratchpad = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * sizeof(int32_t));

        cnn_layer->pwl.pSegments = (intel_pwl_segment_t*)pinned_memory;
        ModelUtilities::GeneratePwlSegments(cnn_layer->pwl.pSegments, cnn_layer->pwl.nSegments);
        pinned_memory += ALIGN64(cnn_layer->pwl.nSegments * sizeof(intel_pwl_segment_t));
    }

    void *pinned_outputs = nullptr;
    if (INTEL_OUTPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_outputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * bytesPerOutput);
    }

    void *pinned_biases = pinned_memory;
    pinned_memory += ALIGN64(nOutputsPerFilter * cnn_layer->nBytesBias);

    void *pinned_filters = pinned_memory;
    pinned_memory += ALIGN64(cnn_layer->nFilters * cnn_layer->nFilterCoefficients * sizeof(int16_t));

    layer->pInputs = pinned_inputs;
    layer->pOutputs = pinned_outputs;
    layer->pOutputsIntermediate = pinned_scratchpad;

    cnn_layer->pBiases = pinned_biases;
    cnn_layer->pFilters = pinned_filters;
}

void ChainModel::setup_rnn_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory)
{
    intel_recurrent_layer_t* recurrent_layer = (intel_recurrent_layer_t*)layer->pLayerStruct;

    void *pinned_inputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_inputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * layer->nInputColumns * sizeof(int16_t));
    }

    void *pinned_outputs = nullptr;
    if (INTEL_OUTPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_outputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * sizeof(int16_t));
    }

    void *pinned_scratchpad = pinned_memory;
    pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * sizeof(int32_t));

    recurrent_layer->pwl.pSegments = (intel_pwl_segment_t*)pinned_memory;
    ModelUtilities::GeneratePwlSegments(recurrent_layer->pwl.pSegments, recurrent_layer->pwl.nSegments);
    pinned_memory += ALIGN64(recurrent_layer->pwl.nSegments * sizeof(intel_pwl_segment_t));

    void *pinned_biases = pinned_memory;
    pinned_memory += ALIGN64(layer->nOutputRows * recurrent_layer->affine.nBytesPerBias);

    void *pinned_weights = pinned_memory;
    pinned_memory += ALIGN64((layer->nOutputRows + layer->nInputRows) * layer->nOutputRows * recurrent_layer->affine.nBytesPerWeight);

    layer->pInputs = pinned_inputs;
    layer->pOutputs = pinned_outputs;
    layer->pOutputsIntermediate = pinned_scratchpad;

    recurrent_layer->affine.pBiases = pinned_biases;
    recurrent_layer->affine.pWeights = pinned_weights;
}

void ChainModel::setup_simple_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory)
{

    void *pinned_inputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_inputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * layer->nInputColumns * sizeof(int16_t));
    }

    void *pinned_outputs = nullptr;
    if (INTEL_OUTPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_outputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nOutputRows * layer->nOutputColumns * sizeof(int16_t));
    }

    layer->pInputs = pinned_inputs;
    layer->pOutputs = pinned_outputs;
    layer->pOutputsIntermediate = nullptr;
}

void ChainModel::setup_gmm_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory)
{
    auto gmm = static_cast<gna_gmm_layer*>(layer->pLayerStruct);

    void *pinned_inputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_inputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * layer->nInputColumns * sizeof(int16_t));
    }

    void *pinned_vars = pinned_memory;
    auto nBytesPerVars = 0;
    if (GNA_MAXMIX16 == gmm->config.mode)
    {
        gmm->data.inverseCovariances.inverseCovariancesForMaxMix8 = nullptr;
        gmm->data.inverseCovariances.inverseCovariancesForMaxMix16 = (uint16_t*)pinned_vars;
        nBytesPerVars = GNA_INT16;
    }
    else
    {
        gmm->data.inverseCovariances.inverseCovariancesForMaxMix8 = (uint8_t*)pinned_vars;
        gmm->data.inverseCovariances.inverseCovariancesForMaxMix16 = nullptr;
        nBytesPerVars = GNA_INT8;
    }
    pinned_memory += ALIGN64(gmm->config.stateCount * gmm->config.mixtureComponentCount * layer->nInputColumns * nBytesPerVars);

    void *pinned_means = pinned_memory;
    pinned_memory += ALIGN64(gmm->config.stateCount * gmm->config.mixtureComponentCount * layer->nInputColumns * sizeof(uint8_t));

    void * pinned_consts = pinned_memory;
    pinned_memory += ALIGN64(gmm->config.stateCount * gmm->config.mixtureComponentCount * sizeof(uint32_t));

    void *pinned_outputs = nullptr;
    if (INTEL_INPUT != layer->mode && INTEL_INPUT_OUTPUT != layer->mode)
    {
        pinned_outputs = pinned_memory;
        pinned_memory += ALIGN64(layer->nInputRows * gmm->config.stateCount * sizeof(int32_t));       // (4 out vectors, 8 elems in each one, 4-byte elems)
    }

    layer->pInputs = pinned_inputs;
    layer->pOutputs = pinned_outputs;

    gmm->data.gaussianConstants = (uint32_t*)pinned_consts;
    gmm->data.meanValues = (uint8_t*)pinned_means;
}

const int ChainModel::groupingNum = 4;
const int ChainModel::inVecSz = 16;
const int ChainModel::cnnInVecSz = 96;
const int ChainModel::cnnOutVecSz = 4;
const int ChainModel::outVecSz = 8;
const int ChainModel::rnnOutVecSz = 32;
const int ChainModel::nSegments = 64;
const int ChainModel::gmmInVecSz = 24;

const int8_t ChainModel::weights_1B[outVecSz * inVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t ChainModel::weights_2B[outVecSz * inVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t ChainModel::inputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4,
    5, -4, -7,  4,
    0,  7,  1, -7,
    1,  6,  7,  9,
    2, -4,  9,  8,
    -5, -1,  2,  9,
    -8, -8,  8,  1,
    -7,  2, -1, -1,
    -9, -5, -8,  5,
    0, -1,  3,  9,
    0,  8,  1, -2,
    -9,  8,  0, -7,
    -9, -8, -1, -4,
    -3, -7, -2,  3,
    -8,  0,  1,  3,
    -4, -6, -8, -2
};

const int16_t ChainModel::cnnInputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4,
    5, -4, -7,  4,
    0,  7,  1, -7,
    1,  6,  7,  9,
    2, -4,  9,  8,
    -5, -1,  2,  9,
    -8, -8,  8,  1,
    -7,  2, -1, -1,
    -9, -5, -8,  5,
    0, -1,  3,  9,
    0,  8,  1, -2,
    -9,  8,  0, -7,
    -9, -8, -1, -4,
    -3, -7, -2,  3,
    -8,  0,  1,  3,
    -4, -6, -8, -2
};

const intel_bias_t ChainModel::regularBiases[outVecSz*groupingNum] = {
    5, 4, -2, 5,
    -7, -5, 4, -1
};

const  intel_compound_bias_t ChainModel::compoundBiases[outVecSz*groupingNum] =
{
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
    {-7,1,{0}}, {-5,1,{0}}, {4,1,{0}}, {-1,1,{0}},
};
