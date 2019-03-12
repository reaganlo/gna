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

#include "SetupConvolutionModel2D.h"

#include "Macros.h"

constexpr uint32_t layerCount = 2;
constexpr uint32_t grouping = 1;

constexpr gna_3d_dimensions inputDimensions = {7, 7, 2};
constexpr uint32_t totalInputCount = inputDimensions.width * inputDimensions.height * inputDimensions.depth;
constexpr gna_3d_dimensions filterDimensions = {3, 3, 2};
constexpr uint32_t filterN = 3;
//constexpr gna_3d_dimensions outputDimensions = {5, 5, filterN};
constexpr gna_3d_dimensions outputDimensions = {3, 3, filterN};
constexpr uint32_t totalOutputCount = outputDimensions.width * outputDimensions.height * outputDimensions.depth;

const int16_t filters[filterN * ALIGN(filterDimensions.width * filterDimensions.height * filterDimensions.depth, (16/sizeof(int16_t)))] =
{
    1,2,    1,2,    1,2,
    1,2,    1,2,    1,2,
    1,2,    1,2,    1,2,
    0,0,0,0,0,0,

    1,2,    1,2,    1,2,
    1,2,    1,2,    1,2,
    1,2,    1,2,    1,2,
    0,0,0,0,0,0,

    1,2,    1,2,    1,2,
    1,2,    1,2,    1,2,
    1,2,    1,2,    1,2,
    0,0,0,0,0,0
};

const int16_t inputs[grouping * totalInputCount] = {
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
   1, 1,    2,2,    3,3,    4,4,    5,5,     6,6,   7,7,
};

const intel_bias_t regularBiases[filterN] = {
    1, 2, 3,
};
//
//const int32_t ref_output[grouping * totalOutputCount] = {
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//};


const int32_t ref_output[grouping * totalOutputCount] = {
   1,1,1,    2,2,2,    3,3,3,
   1,1,1,    2,2,2,    3,3,3,
   1,1,1,    2,2,2,    3,3,3,
};

SetupConvolutionModel2D::SetupConvolutionModel2D(DeviceController & deviceCtrl, bool pwlEn)
    : ModelSetup{deviceCtrl, {layerCount, grouping, nullptr}, ref_output},
    pwlEnabled{pwlEn}
{
    sampleConvolutionLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, InputComponent, 0, inputBuffer);
    deviceController.BufferAdd(configId, OutputComponent, 0, outputBuffer);
}

SetupConvolutionModel2D::~SetupConvolutionModel2D()
{
   deviceController.Free(memory);
}

void SetupConvolutionModel2D::sampleConvolutionLayer()
{
    uint32_t buf_size_filters = ALIGN64(sizeof(filters));
    uint32_t buf_size_inputs = ALIGN64(sizeof(inputs));
    uint32_t buf_size_biases = ALIGN64(sizeof(regularBiases));
    uint32_t buf_size_outputs = ALIGN64(sizeof(ref_output));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs;
    uint32_t bytes_granted;
    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    void* pinned_filters = pinned_mem_ptr;
    memcpy(pinned_filters, filters, sizeof(filters));
    pinned_mem_ptr += buf_size_filters;

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    pinned_mem_ptr += buf_size_biases;

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;
    outputBuffer = pinned_mem_ptr;

    layer.activation.nSegments = 0;
    layer.activation.pSegments = nullptr;
    layer.convolution.biases.biasesData = pinned_biases;
    layer.convolution.biases.dataMode = GNA_INT32;
    layer.convolution.biases.mode = GNA_BIAS_PER_KERNEL;
    layer.convolution.filters.count = filterN;
    layer.convolution.filters.dataMode = GNA_INT16;
    layer.convolution.filters.dimensions = filterDimensions;
    layer.convolution.filters.filtersData =  (void*)pinned_filters;
    layer.convolution.stride = {1, 1, 0};
    layer.convolution.zeroPadding = {};
    layer.inputDimensions = inputDimensions;
    layer.pooling.type = INTEL_MAX_POOLING;
    layer.pooling.stride = {1, 1, 0};
    layer.pooling.window = {3, 3, 0};


    nnet.pLayers[0].nInputColumns = totalInputCount;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = totalOutputCount;
    nnet.pLayers[0].nOutputRows = nnet.nGroup;
    nnet.pLayers[0].pOutputsIntermediate = nullptr;
    nnet.pLayers[0].nBytesPerOutput = GNA_INT32;
    nnet.pLayers[0].nBytesPerInput = GNA_INT16;
    nnet.pLayers[0].nBytesPerIntermediateOutput = GNA_INT32;
    nnet.pLayers[0].operation = INTEL_CONVOLUTIONAL_2D;
    nnet.pLayers[0].mode = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;

    memcpy_s(&nnet.pLayers[1], sizeof(intel_nnet_layer_t), &nnet.pLayers[0], sizeof(intel_nnet_layer_t));
}
