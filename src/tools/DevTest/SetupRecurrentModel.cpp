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

#include "SetupRecurrentModel.h"

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupRecurrentModel::SetupRecurrentModel(DeviceController & deviceCtrl, bool wght2B)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{wght2B}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleRnnLayer(nnet);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, InputComponent, 0, inputBuffer);
    deviceController.BufferAdd(configId, OutputComponent, 0, outputBuffer);
}

SetupRecurrentModel::~SetupRecurrentModel()
{
    deviceController.Free(memory);
    free(nnet.pLayers);

    deviceController.ModelRelease(modelId);
}

void SetupRecurrentModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    UNREFERENCED_PARAMETER(configIndex);
    for (unsigned int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupRecurrentModel::samplePwl(intel_pwl_segment_t *segments, uint32_t numberOfSegments)
{
    auto xBase = -200;
    auto xBaseInc = 2*abs(xBase) / numberOfSegments;
    auto yBase = -200;
    auto yBaseInc = 1;
    for (auto i = uint32_t{0}; i < numberOfSegments; i++, xBase += xBaseInc, yBase += yBaseInc, yBaseInc++)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = static_cast<int16_t>(yBase);
        segments[i].slope = 1;
    }
}

void SetupRecurrentModel::sampleRnnLayer(intel_nnet_type_t& hNnet)
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_scratchpad = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int16_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_scratchpad + buf_size_outputs + buf_size_tmp_outputs + buf_size_pwl;
    uint32_t bytes_granted;

    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

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

    scratchpad = pinned_mem_ptr;
    memset(scratchpad, 0, buf_size_scratchpad);
    pinned_mem_ptr += buf_size_scratchpad;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    affine_func.nBytesPerWeight = weightsAre2Bytes ? GNA_INT16 : GNA_INT8;
    affine_func.nBytesPerBias = weightsAre2Bytes ? GNA_INT32: GNA_DATA_RICH_FORMAT;
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    pwl.nSegments = nSegments;
    pwl.pSegments = reinterpret_cast<intel_pwl_segment_t*>(pinned_mem_ptr);
    samplePwl(pwl.pSegments, pwl.nSegments);
    pinned_mem_ptr += buf_size_pwl;

    recurrent_layer.affine = affine_func;
    recurrent_layer.pwl = pwl;
    recurrent_layer.feedbackFrameDelay = 3;

    hNnet.pLayers[0].nInputColumns = inVecSz;
    hNnet.pLayers[0].nInputRows = hNnet.nGroup;
    hNnet.pLayers[0].nOutputColumns = outVecSz;
    hNnet.pLayers[0].nOutputRows = hNnet.nGroup;
    hNnet.pLayers[0].nBytesPerInput = GNA_INT16;
    hNnet.pLayers[0].nBytesPerOutput = GNA_INT16;
    hNnet.pLayers[0].nBytesPerIntermediateOutput = GNA_INT32;
    hNnet.pLayers[0].operation = INTEL_RECURRENT;
    hNnet.pLayers[0].mode = INTEL_INPUT_OUTPUT;
    hNnet.pLayers[0].pLayerStruct = &recurrent_layer;
    hNnet.pLayers[0].pInputs = nullptr;
    hNnet.pLayers[0].pOutputsIntermediate = scratchpad;
    hNnet.pLayers[0].pOutputs = nullptr;
}
