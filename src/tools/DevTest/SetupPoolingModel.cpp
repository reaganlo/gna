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

namespace
{
const int layersNum = 1;
const int groupingNum = 1;
const int nFilters = 4;
const int nFilterCoefficients = 48;
const int inVecSz = 96;
const int outVecSz = 4;

const int16_t filters[nFilters * nFilterCoefficients] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,

    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,

    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7,
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,

    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
};

const int16_t inputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
    2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
    -9, -5, -8,  5, 0, -1,  3,  9, 0,  8,  1, -2, -9,  8,  0, -7,

    -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2,
    -5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
    2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1
};

const intel_bias_t regularBiases[outVecSz*groupingNum] = {
    5, 4, -2, 5
};

const  intel_compound_bias_t compoundBiases[outVecSz*groupingNum] =
{
    { 5,1,{0} },
    {4,1,{0}},
    {-2,1,{0}},
    {5,1,{0}},
};

const int16_t ref_output[outVecSz * groupingNum] =
{
    1170, -410, -39, 1230
};
}

SetupPoolingModel::SetupPoolingModel(DeviceController & deviceCtrl)
    : deviceController{deviceCtrl}
{
    uint32_t nSegments = 64;
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    samplePoolingLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupPoolingModel::~SetupPoolingModel()
{
    deviceController.ModelRelease(modelId);
    deviceController.Free();

    free(nnet.pLayers);
}

void SetupPoolingModel::checkReferenceOutput() const
{
    for (int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            throw std::exception("Wrong output");
        }
    }
}

void SetupPoolingModel::samplePwl(intel_pwl_segment_t *segments, uint32_t nSegments)
{
    auto xBase = -600;
    auto xBaseInc = 2*abs(xBase) / nSegments;
    auto yBase = xBase;
    auto yBaseInc = 1;
    for (auto i = 0ui32; i < nSegments; i++, xBase += xBaseInc, yBase += yBaseInc, yBaseInc++) 
    {
        segments[i].xBase = xBase;
        segments[i].yBase = yBase;
        segments[i].slope = 1;
    }
}

void SetupPoolingModel::samplePoolingLayer()
{
    int buf_size_filters = ALIGN64(sizeof(filters));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = ALIGN64(sizeof(regularBiases));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int16_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs + buf_size_pwl;
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, &bytes_granted);

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
    pinned_mem_ptr += nSegments * sizeof(intel_pwl_segment_t);

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
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t); // activated
    nnet.pLayers[0].nBytesPerIntermediateOutput = sizeof(int32_t);
    nnet.pLayers[0].nLayerKind = INTEL_CONVOLUTIONAL;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &convolution_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
    nnet.pLayers[0].pOutputs = nullptr;
}

