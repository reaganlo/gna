/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "ModelUtilities.h"
#include "SetupSplitModel.h"

#include <cstring>
#include <iostream>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupSplitModel::SetupSplitModel(DeviceController & deviceCtrl, bool weight2B, bool activeListEn, bool pwlEn)
    : deviceController{ deviceCtrl },
    weightsAre2Bytes{ weight2B },
    pwlEnabled{ pwlEn }
{
    UNREFERENCED_PARAMETER(activeListEn);

    auto firstModelSize = getFirstModelSize();
    auto secondModelSize = getSecondModelSize();

    auto inputsSize = 2 * ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(inputs.at(0).at(0).size() * sizeof(int16_t)));
    auto outputsSize = 2 * ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(affineOutputs.at(0).size() * sizeof(int32_t)));
    auto firstModelConfigSize = inputsSize + outputsSize;

    inputsSize = 2 * ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(inputs.at(1).at(0).size() * sizeof(int16_t)));
    outputsSize = 2 * ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(diagonalOutputs.at(0).size() * sizeof(int32_t)));
    auto secondModelConfigSize = inputsSize + outputsSize;

    auto wholeSize = firstModelSize + firstModelConfigSize + secondModelSize + secondModelConfigSize;

    auto grantedSize = uint32_t{0};
    memory = deviceController.Alloc(static_cast<uint32_t>(wholeSize), &grantedSize);
    auto pinned_memory = static_cast<uint8_t*>(memory);

    setupFirstAffineLayer(pinned_memory);
    setupSecondAffineLayer(pinned_memory);

    uint32_t modelIdSplit;

    deviceController.ModelCreate(&firstModel, &modelIdSplit);
    models.push_back(modelIdSplit);

    configId = deviceController.ConfigAdd(modelIdSplit);
    modelsConfigurations[modelIdSplit].push_back(configId);

    configId = deviceController.ConfigAdd(modelIdSplit);
    modelsConfigurations[modelIdSplit].push_back(configId);

    setupInputBuffer(pinned_memory, 0, 0);
    setupInputBuffer(pinned_memory, 0, 1);
    setupOutputBuffer(pinned_memory, 0, 0);
    setupOutputBuffer(pinned_memory, 0, 1);

    deviceController.ModelCreate(&secondModel, &modelIdSplit);
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
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

void SetupSplitModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
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

void SetupSplitModel::setupFirstAffineLayer(uint8_t*& pinned_memory)
{
    const auto buf_size_weights = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(weights_2B)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(weights_1B)));
    const auto buf_size_biases = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(regularBiases)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(compoundBiases)));
    void* pinned_weights = pinned_memory;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_weights, weights_2B, sizeof(weights_2B));
    }
    else
    {
        memcpy(pinned_weights, weights_1B, sizeof(weights_1B));
    }
    pinned_memory += buf_size_weights;


    int32_t* pinned_biases = (int32_t*)pinned_memory;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    }
    else
    {
        memcpy(pinned_biases, compoundBiases, sizeof(compoundBiases));
    }
    pinned_memory += buf_size_biases;

    operationHolder.InitAffineEx(inVecSz, outVecSz, groupingNum, nullptr, nullptr,
        pinned_weights, weightsAre2Bytes ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        pinned_biases, weightsAre2Bytes ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias);

    if (pwlEnabled)
    {
        Gna2PwlSegment* pinned_pwl = reinterpret_cast<Gna2PwlSegment*>(pinned_memory);

        ModelUtilities::GeneratePwlSegments(pinned_pwl, nSegments);
        operationHolder.AddPwl(nSegments, pinned_pwl, Gna2DataTypeInt16);
    }

    firstModel = { 1, &operationHolder.Get() };
}

void SetupSplitModel::setupSecondAffineLayer(uint8_t*& pinned_memory)
{
    const auto buf_size_weights = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(diagonal_weights_2B)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(diagonal_weights_1B)));
    const auto buf_size_biases = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(diagonalRegularBiases)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(diagonalCompoundBiases)));

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

    int32_t* pinned_biases = (int32_t*)pinned_memory;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, diagonalRegularBiases, sizeof(diagonalRegularBiases));
    }
    else
    {
        memcpy(pinned_biases, diagonalCompoundBiases, sizeof(diagonalCompoundBiases));
    }
    pinned_memory += buf_size_biases;

    operation2Holder.InitDiagonalEx(inVecSz, outVecSizeDiagonal, groupingNum, nullptr, nullptr,
        pinned_weights, weightsAre2Bytes ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        pinned_biases, weightsAre2Bytes ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias);

    if (pwlEnabled)
    {
        Gna2PwlSegment* pinned_pwl = reinterpret_cast<Gna2PwlSegment*>(pinned_memory);

        ModelUtilities::GeneratePwlSegments(pinned_pwl, nSegments);
        operation2Holder.AddPwl(nSegments, pinned_pwl, Gna2DataTypeInt16);
    }
    secondModel = { 1, &operation2Holder.Get() };
}

size_t SetupSplitModel::getFirstModelSize()
{
    uint32_t buf_size_weights = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(weights_2B)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(weights_1B)));
    uint32_t buf_size_biases = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(regularBiases)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(compoundBiases)));
    uint32_t buf_size_tmp_outputs = ModelUtilities::CastAndRoundUpTo64(
            outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int32_t)));
    uint32_t buf_size_pwl = ModelUtilities::CastAndRoundUpTo64(
            nSegments * static_cast<uint32_t>(sizeof(Gna2PwlSegment)));

    uint32_t bytes_requested = buf_size_weights + buf_size_biases + buf_size_tmp_outputs;

    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }

    return bytes_requested;
}

size_t SetupSplitModel::getSecondModelSize()
{
    uint32_t buf_size_weights = static_cast<uint32_t>(weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(sizeof(weights_2B))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(weights_1B))));
    uint32_t buf_size_biases = static_cast<uint32_t>(weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(sizeof(regularBiases)) : ModelUtilities::CastAndRoundUpTo64(sizeof(compoundBiases)));
    uint32_t buf_size_tmp_outputs = static_cast<uint32_t>(ModelUtilities::CastAndRoundUpTo64(
        diagonalOutVecSz * groupingNum * sizeof(int32_t)));
    uint32_t buf_size_pwl = static_cast<uint32_t>(
        ModelUtilities::CastAndRoundUpTo64(nSegments * sizeof(Gna2PwlSegment)));

    uint32_t bytes_requested = buf_size_weights + buf_size_biases + buf_size_tmp_outputs;
    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }

    return bytes_requested;
}

void SetupSplitModel::setupInputBuffer(uint8_t* &pinned_memory, uint32_t modelIndex, uint32_t configIndex)
{
    modelId = models.at(modelIndex);

    configId = modelsConfigurations.at(modelId).at(configIndex);

    auto& pinnedInput = configurationBuffers[modelId][configIndex].first;
    pinnedInput = pinned_memory;
    auto& srcBuffer = inputs.at(modelIndex).at(configIndex);

    const auto inputsSize = static_cast<uint32_t>(srcBuffer.size() * sizeof(int16_t));
    memcpy(pinnedInput, srcBuffer.data(), inputsSize);
    DeviceController::BufferAdd(configId, 0, InputOperandIndex, pinnedInput);

    auto buf_size_inputs = ModelUtilities::CastAndRoundUpTo64(inputsSize);
    pinned_memory += buf_size_inputs;
}

void SetupSplitModel::setupOutputBuffer(uint8_t* &pinned_memory, uint32_t modelIndex, uint32_t configIndex)
{
    auto modelIdSplit= models.at(modelIndex);
    configId = modelsConfigurations.at(modelIdSplit).at(configIndex);

    auto& pinnedOutput = configurationBuffers[modelIdSplit][configId].second;
    pinnedOutput = pinned_memory;
    DeviceController::BufferAdd(configId, 0, OutputOperandIndex, pinnedOutput);

    auto outputsSize = groupingNum * ((0 == modelIndex) ? outVecSz : diagonalOutVecSz);
    outputsSize *= static_cast<uint32_t>(pwlEnabled ? sizeof(int16_t) : sizeof(int32_t));
    auto buf_size_outputs = ModelUtilities::CastAndRoundUpTo64(outputsSize);
    pinned_memory += buf_size_outputs;
}
