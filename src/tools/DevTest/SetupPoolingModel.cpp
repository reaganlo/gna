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

#include "SetupPoolingModel.h"

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupPoolingModel::SetupPoolingModel(DeviceController & deviceCtrl)
    : deviceController{deviceCtrl}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    samplePoolingLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupPoolingModel::~SetupPoolingModel()
{
    deviceController.Free(memory);
    free(nnet.pLayers);

    deviceController.ModelRelease(modelId);
}

void SetupPoolingModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    UNREFERENCED_PARAMETER(configIndex);
    for (unsigned int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupPoolingModel::samplePwl(intel_pwl_segment_t *segments, uint32_t numberOfSegments)
{
    auto xBase = -600;
    auto xBaseInc = 2u * static_cast<uint32_t>(abs(xBase)) / numberOfSegments;
    auto yBase = xBase;
    auto yBaseInc = 1;
    for (auto i = uint32_t{0}; i < numberOfSegments; i++, xBase += xBaseInc, yBase += yBaseInc, yBaseInc++)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = static_cast<int16_t>(yBase);
        segments[i].slope = 1;
    }
}

void SetupPoolingModel::samplePoolingLayer()
{
    uint32_t buf_size_filters = ALIGN64(static_cast<uint32_t>(sizeof(filters)));
    uint32_t buf_size_inputs = ALIGN64(static_cast<uint32_t>(sizeof(inputs)));
    uint32_t buf_size_biases = ALIGN64(static_cast<uint32_t>(sizeof(regularBiases)));
    uint32_t buf_size_outputs = ALIGN64(
            outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int16_t)));
    uint32_t buf_size_tmp_outputs = ALIGN64(
            outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int32_t)));
    uint32_t buf_size_pwl = ALIGN64(
            nSegments * static_cast<uint32_t>(sizeof(intel_pwl_segment_t)));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs + buf_size_pwl;
    uint32_t bytes_granted;

    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    void* pinned_filters = pinned_mem_ptr;
    memcpy(pinned_filters, filters, sizeof(filters));
    pinned_mem_ptr += buf_size_filters;

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    pinned_mem_ptr += buf_size_biases;

    void *tmp_outputs = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_tmp_outputs;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    void* pinned_pwl = pinned_mem_ptr;

    pwl.nSegments = nSegments;
    pwl.pSegments = reinterpret_cast<intel_pwl_segment_t*>(pinned_pwl);
    samplePwl(pwl.pSegments, pwl.nSegments);

    convolution_layer.nBytesBias = sizeof(intel_bias_t);
    convolution_layer.pBiases = pinned_biases;
    convolution_layer.pwl = pwl;
    convolution_layer.nBytesFilterCoefficient = sizeof(int16_t);
    convolution_layer.nFeatureMaps = 1;
    convolution_layer.nFeatureMapRows = 1;
    convolution_layer.nFeatureMapColumns = 48;
    convolution_layer.pFilters = pinned_filters;
    convolution_layer.nFilters = 4;
    convolution_layer.nFilterRows = 1;
    convolution_layer.nFilterCoefficients = 48;
    convolution_layer.poolType = INTEL_SUM_POOLING;
    convolution_layer.nPoolSize = 6;
    convolution_layer.nPoolStride = 6;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = outVecSz;
    nnet.pLayers[0].nOutputRows = nnet.nGroup;
    nnet.pLayers[0].nBytesPerInput = GNA_INT16;
    nnet.pLayers[0].nBytesPerOutput = GNA_INT16; // activated
    nnet.pLayers[0].nBytesPerIntermediateOutput = GNA_INT32;
    nnet.pLayers[0].operation = INTEL_CONVOLUTIONAL;
    nnet.pLayers[0].mode = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &convolution_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
    nnet.pLayers[0].pOutputs = nullptr;
}

