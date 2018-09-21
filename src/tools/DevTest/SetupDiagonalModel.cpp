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

#include "SetupDiagonalModel.h"

SetupDiagonalModel::SetupDiagonalModel(DeviceController & deviceCtrl, bool wght2B, bool pwlEn)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{wght2B},
    pwlEnabled{pwlEn}
{
    sampleAffineLayer(nnet);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupDiagonalModel::~SetupDiagonalModel()
{
    deviceController.Free();
}

template <class intel_reference_output_type>
intel_reference_output_type* SetupDiagonalModel::refOutputAssign(int configIndex) const
{
    switch (configIndex)
    {
    case configDiagonal_1_1B:
        return (intel_reference_output_type*)ref_output;
    case confiDiagonal_1_2B:
        return (intel_reference_output_type*)ref_output;
    case confiDiagonalPwl_1_1B:
        return (intel_reference_output_type*)ref_output_pwl;
    case confiDiagonalPwl_1_2B:
        return (intel_reference_output_type*)ref_output_pwl;
    default:
        throw std::runtime_error("Invalid configuration index");
    }
}

template <class intel_reference_output_type>
void SetupDiagonalModel::compareReferenceValues(unsigned int i, int configIndex) const
{
    intel_reference_output_type outElemVal = static_cast<const intel_reference_output_type*>(outputBuffer)[i];
    const intel_reference_output_type* refOutput = refOutputAssign<intel_reference_output_type>(configIndex);
    if (refOutput[i] != outElemVal)
    {
        // TODO: how it should notified? return or throw
        throw std::runtime_error("Wrong output");
    }
}


void SetupDiagonalModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    unsigned int ref_output_size = refSize[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        (configIndex == configPwl) ? compareReferenceValues<int16_t>(i, configIndex) : compareReferenceValues<int16_t>(i, configIndex);
    }
}

void SetupDiagonalModel::sampleAffineLayer(intel_nnet_type_t& hNnet)
{
    hNnet.nGroup = groupingNum;
    hNnet.nLayers = layersNum;
    hNnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    if (pwlEnabled) bytes_requested += buf_size_pwl;
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, hNnet.nLayers, 0, &bytes_granted);

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

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

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

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    void *tmp_outputs = nullptr;
    if (pwlEnabled)
    {
        tmp_outputs = pinned_mem_ptr;
        pinned_mem_ptr += buf_size_tmp_outputs;

        intel_pwl_segment_t *pinned_pwl = reinterpret_cast<intel_pwl_segment_t*>(pinned_mem_ptr);
        pinned_mem_ptr += buf_size_pwl;

        pwl.nSegments = nSegments;
        pwl.pSegments = pinned_pwl;
        samplePwl(pwl.pSegments, pwl.nSegments);
    }
    else
    {
        pwl.nSegments = 0;
        pwl.pSegments = nullptr;
    }

    affine_func.nBytesPerWeight = weightsAre2Bytes ? 2 : 1;
    affine_func.nBytesPerBias = weightsAre2Bytes ? sizeof(intel_bias_t) : sizeof(intel_compound_bias_t);
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    hNnet.pLayers[0].nInputColumns = hNnet.nGroup;
    hNnet.pLayers[0].nInputRows = inVecSz;
    hNnet.pLayers[0].nOutputColumns = hNnet.nGroup;
    hNnet.pLayers[0].nOutputRows = outVecSz;
    hNnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    if (pwlEnabled)
    {
        hNnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
        hNnet.pLayers[0].nBytesPerOutput = sizeof(int16_t);
    }
    else
    {
        hNnet.pLayers[0].pOutputsIntermediate = nullptr;
        hNnet.pLayers[0].nBytesPerOutput = sizeof(int32_t);
    }
    hNnet.pLayers[0].nBytesPerIntermediateOutput = 4;
    hNnet.pLayers[0].nLayerKind = INTEL_AFFINE_DIAGONAL;
    hNnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    hNnet.pLayers[0].pLayerStruct = &affine_layer;
    hNnet.pLayers[0].pInputs = nullptr;
    hNnet.pLayers[0].pOutputs = nullptr;
}

void SetupDiagonalModel::samplePwl(intel_pwl_segment_t *segments, uint32_t numberOfSegments)
{
    auto xBase = INT32_MIN;
    auto xBaseInc = UINT32_MAX / numberOfSegments;
    auto yBase = INT32_MAX;
    auto yBaseInc = UINT16_MAX / numberOfSegments;
    for (auto i = uint32_t{0}; i < numberOfSegments; i++, xBase += xBaseInc, yBase += yBaseInc)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = yBase;
        segments[i].slope = 1;
    }
}
