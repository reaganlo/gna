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

#include <cstring>
#include <cstdlib>
#include <iostream>

#include "gna-api.h"

#include "ModelUtilities.h"
#include "SetupSplitModel.h"

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupSplitModel::SetupSplitModel(DeviceController & deviceCtrl, bool wght2B, bool activeListEn, bool pwlEn)
    : deviceController{ deviceCtrl },
    weightsAre2Bytes{ wght2B },
    pwlEnabled{ pwlEn }
{
    UNREFERENCED_PARAMETER(activeListEn);

    firstNnet.nGroup = groupingNum;
    firstNnet.nLayers = 1;
    firstNnet.pLayers = (intel_nnet_layer_t*)calloc(firstNnet.nLayers, sizeof(intel_nnet_layer_t));

    secondNnet.nGroup = groupingNum;
    secondNnet.nLayers = 1;
    secondNnet.pLayers = (intel_nnet_layer_t*)calloc(secondNnet.nLayers, sizeof(intel_nnet_layer_t));

    auto firstModelSize = getFirstModelSize();
    auto secondModelSize = getSecondModelSize();

    auto inputsSize = 2 * ALIGN64(inputs.at(0).at(0).size() * sizeof(int16_t));
    auto outputsSize = 2 * ALIGN64(affineOutputs.at(0).size() * sizeof(int32_t));
    auto firstModelConfigSize = inputsSize + outputsSize;

    inputsSize = 2 * ALIGN64(inputs.at(1).at(0).size() * sizeof(int16_t));
    outputsSize = 2 * ALIGN64(diagonalOutputs.at(0).size() * sizeof(int32_t));
    auto secondModelConfigSize = inputsSize + outputsSize;

    auto wholeSize = firstModelSize + firstModelConfigSize + secondModelSize + secondModelConfigSize;

    auto totalLayerCount = firstNnet.nLayers + secondNnet.nLayers;
    auto totalGmmCount = uint16_t{0};

    auto grantedSize = uint32_t{0};
    auto pinned_memory = deviceController.Alloc(static_cast<uint32_t>(wholeSize), static_cast<uint16_t>(totalLayerCount), totalGmmCount, &grantedSize);
    if (NULL == pinned_memory || grantedSize < wholeSize)
    {
        throw GNA_ERR_RESOURCES;
    }

    setupFirstAffineLayer(pinned_memory);
    setupSecondAffineLayer(pinned_memory);

    gna_model_id modelIdSplit;

    deviceController.ModelCreate(&firstNnet, &modelIdSplit);
    models.push_back(modelIdSplit);

    configId = deviceController.ConfigAdd(modelIdSplit);
    modelsConfigurations[modelIdSplit].push_back(configId);

    configId = deviceController.ConfigAdd(modelIdSplit);
    modelsConfigurations[modelIdSplit].push_back(configId);

    setupInputBuffer(pinned_memory, 0, 0);
    setupInputBuffer(pinned_memory, 0, 1);
    setupOutputBuffer(pinned_memory, 0, 0);
    setupOutputBuffer(pinned_memory, 0, 1);

    deviceController.ModelCreate(&secondNnet, &modelIdSplit);
    models.push_back(modelIdSplit);

    configId = deviceController.ConfigAdd(modelIdSplit);
    modelsConfigurations[modelIdSplit].push_back(configId);

    configId = deviceController.ConfigAdd(modelIdSplit);
    modelsConfigurations[modelIdSplit].push_back(configId);

    setupInputBuffer(pinned_memory, 1, 0);
    setupInputBuffer(pinned_memory, 1, 1);
    setupOutputBuffer(pinned_memory, 1, 0);
    setupOutputBuffer(pinned_memory, 1, 1);
}

SetupSplitModel::~SetupSplitModel()
{
    deviceController.Free();
    free(firstNnet.pLayers);
    free(secondNnet.pLayers);
}

void SetupSplitModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    std::cout << "(model, configuration) " << modelIndex << " " << configIndex << ": ";
    auto outputCount = (0 == modelIndex) ? affineOutputs.at(configIndex).size() : diagonalOutputs.at(configIndex).size();
    auto refOutputs = (0 == modelIndex) ? affineOutputs.at(configIndex).data() : diagonalOutputs.at(configIndex).data();

    auto modelIdSplit = models.at(modelIndex);
    auto configIdSplit = modelsConfigurations.at(modelIdSplit).at(configIndex);
    auto outputBuffer = static_cast<int32_t*>(configurationBuffers.at(modelIdSplit).at(configIdSplit).second);

    for (unsigned int i = 0; i < outputCount; ++i)
    {
        int32_t outElemVal = outputBuffer[i];
        if (refOutputs[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupSplitModel::setupFirstAffineLayer(uint8_t* &pinned_mem_ptr)
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    void* pinned_weights = pinned_mem_ptr;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_weights, weights_2B, sizeof(weights_2B));
    }
    else
    {
        memcpy(pinned_weights, weights_1B, sizeof(weights_1B));
    }
    pinned_mem_ptr += buf_size_weights;


    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    }
    else
    {
        memcpy(pinned_biases, compoundBiases, sizeof(compoundBiases));
    }
    pinned_mem_ptr += buf_size_biases;

    void *tmp_outputs = nullptr;
    if (pwlEnabled)
    {
        tmp_outputs = pinned_mem_ptr;
        pinned_mem_ptr += buf_size_tmp_outputs;

        intel_pwl_segment_t *pinned_pwl = reinterpret_cast<intel_pwl_segment_t*>(pinned_mem_ptr);
        pinned_mem_ptr += buf_size_pwl;

        firstPwl.nSegments = nSegments;
        firstPwl.pSegments = pinned_pwl;
        ModelUtilities::GeneratePwlSegments(firstPwl.pSegments, firstPwl.nSegments);
    }
    else
    {
        firstPwl.nSegments = 0;
        firstPwl.pSegments = nullptr;
    }

    firstAffineFunc.nBytesPerWeight = weightsAre2Bytes ? GNA_INT16 : GNA_INT8;
    firstAffineFunc.nBytesPerBias = weightsAre2Bytes ? GNA_INT32: GNA_DATA_RICH_FORMAT;
    firstAffineFunc.pWeights = pinned_weights;
    firstAffineFunc.pBiases = pinned_biases;

    firstAffineLayer.affine = firstAffineFunc;
    firstAffineLayer.pwl = firstPwl;

    firstNnet.pLayers[0].nInputColumns = firstNnet.nGroup;
    firstNnet.pLayers[0].nInputRows = inVecSz;
    firstNnet.pLayers[0].nOutputColumns = firstNnet.nGroup;
    firstNnet.pLayers[0].nOutputRows = outVecSz;
    firstNnet.pLayers[0].nBytesPerInput = GNA_INT16;
    firstNnet.pLayers[0].nBytesPerIntermediateOutput = GNA_INT32;
    firstNnet.pLayers[0].operation = INTEL_AFFINE;
    firstNnet.pLayers[0].mode = INTEL_INPUT_OUTPUT;
    firstNnet.pLayers[0].pLayerStruct = &firstAffineLayer;
    firstNnet.pLayers[0].pInputs = nullptr;
    firstNnet.pLayers[0].pOutputs = nullptr;

    if (pwlEnabled)
    {
        firstNnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
        firstNnet.pLayers[0].nBytesPerOutput = GNA_INT16;
    }
    else
    {
        firstNnet.pLayers[0].pOutputsIntermediate = nullptr;
        firstNnet.pLayers[0].nBytesPerOutput = GNA_INT32;
    }
}

void SetupSplitModel::setupSecondAffineLayer(uint8_t* &pinned_memory)
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(diagonal_weights_2B)) : ALIGN64(sizeof(diagonal_weights_1B));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(diagonalRegularBiases)) : ALIGN64(sizeof(diagonalCompoundBiases));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    void* pinned_weights = pinned_memory;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_weights, diagonal_weights_2B, sizeof(diagonal_weights_2B));
    }
    else
    {
        memcpy(pinned_weights, diagonal_weights_1B, sizeof(diagonal_weights_1B));
    }
    pinned_memory += buf_size_weights;

    int32_t *pinned_biases = (int32_t*)pinned_memory;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, diagonalRegularBiases, sizeof(diagonalRegularBiases));
    }
    else
    {
        memcpy(pinned_biases, diagonalCompoundBiases, sizeof(diagonalCompoundBiases));
    }
    pinned_memory += buf_size_biases;

    void *tmp_outputs = nullptr;
    if (pwlEnabled)
    {
        tmp_outputs = pinned_memory;
        pinned_memory += buf_size_tmp_outputs;

        intel_pwl_segment_t *pinned_pwl = reinterpret_cast<intel_pwl_segment_t*>(pinned_memory);
        pinned_memory += buf_size_pwl;

        secondPwl.nSegments = nSegments;
        secondPwl.pSegments = pinned_pwl;
        ModelUtilities::GeneratePwlSegments(secondPwl.pSegments, secondPwl.nSegments);
    }
    else
    {
        secondPwl.nSegments = 0;
        secondPwl.pSegments = nullptr;
    }

    secondAffineFunc.nBytesPerWeight = weightsAre2Bytes ? GNA_INT16 : GNA_INT8;
    secondAffineFunc.nBytesPerBias = weightsAre2Bytes ? GNA_INT32: GNA_DATA_RICH_FORMAT;
    secondAffineFunc.pWeights = pinned_weights;
    secondAffineFunc.pBiases = pinned_biases;

    secondAffineLayer.affine = secondAffineFunc;
    secondAffineLayer.pwl = secondPwl;

    secondNnet.pLayers[0].nInputColumns = secondNnet.nGroup;
    secondNnet.pLayers[0].nInputRows = diagonalInVecSz;
    secondNnet.pLayers[0].nOutputColumns = secondNnet.nGroup;
    secondNnet.pLayers[0].nOutputRows = diagonalOutVecSz;
    secondNnet.pLayers[0].nBytesPerInput = GNA_INT16;
    if (pwlEnabled)
    {
        secondNnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
        secondNnet.pLayers[0].nBytesPerOutput = GNA_INT16;
    }
    else
    {
        secondNnet.pLayers[0].pOutputsIntermediate = nullptr;
        secondNnet.pLayers[0].nBytesPerOutput = GNA_INT32;
    }
    secondNnet.pLayers[0].nBytesPerIntermediateOutput = GNA_INT32;
    secondNnet.pLayers[0].operation = INTEL_AFFINE_DIAGONAL;
    secondNnet.pLayers[0].mode = INTEL_INPUT_OUTPUT;
    secondNnet.pLayers[0].pLayerStruct = &secondAffineLayer;
    secondNnet.pLayers[0].pInputs = nullptr;
    secondNnet.pLayers[0].pOutputs = nullptr;
}

size_t SetupSplitModel::getFirstModelSize()
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_biases + buf_size_tmp_outputs;

    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }

    return bytes_requested;
}

size_t SetupSplitModel::getSecondModelSize()
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_tmp_outputs = ALIGN64(diagonalOutVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_biases + buf_size_tmp_outputs;
    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }

    return bytes_requested;
}

void SetupSplitModel::setupInputBuffer(uint8_t* &pinned_memory, int modelIndex, int configIndex)
{
    modelId = models.at(modelIndex);

    configId = modelsConfigurations.at(modelId).at(configIndex);

    auto& pinnedInput = configurationBuffers[modelId][configIndex].first;
    pinnedInput = pinned_memory;
    auto& srcBuffer = inputs.at(modelIndex).at(configIndex);

    auto inputsSize = srcBuffer.size() * sizeof(int16_t);
    memcpy(pinnedInput, srcBuffer.data(), inputsSize);
    deviceController.BufferAdd(configId, InputComponent, 0, pinnedInput);

    auto buf_size_inputs = ALIGN64(inputsSize);
    pinned_memory += buf_size_inputs;
}

void SetupSplitModel::setupOutputBuffer(uint8_t* &pinned_memory, int modelIndex, int configIndex)
{
    auto modelIdSplit= models.at(modelIndex);
    configId = modelsConfigurations.at(modelIdSplit).at(configIndex);

    auto& pinnedOutput = configurationBuffers[modelIdSplit][configId].second;
    pinnedOutput = pinned_memory;
    deviceController.BufferAdd(configId, OutputComponent, 0, pinnedOutput);

    auto outputsSize = groupingNum * ((0 == modelIndex) ? outVecSz : diagonalOutVecSz);
    outputsSize *= (pwlEnabled ? sizeof(int16_t) : sizeof(int32_t));
    auto buf_size_outputs = ALIGN64(outputsSize);
    pinned_memory += buf_size_outputs;
}
