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

#include "SetupConvolutionModel.h"

SetupConvolutionModel::SetupConvolutionModel(DeviceController & deviceCtrl, bool pwlEn)
    : deviceController{deviceCtrl},
      pwlEnabled{pwlEn}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleConvolutionLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupConvolutionModel::~SetupConvolutionModel()
{
    deviceController.Free();

    free(nnet.pLayers);
}

template <class intel_reference_output_type>
intel_reference_output_type* SetupConvolutionModel::refOutputAssign(int configIndex) const
{
    if (configIndex == configPwlEnabled)
    {
        return (intel_reference_output_type*)ref_outputPwl;
    }
    return (intel_reference_output_type*)ref_output;
}

template <class intel_reference_output_type>
void SetupConvolutionModel::compareReferenceValues(unsigned i, int configIndex) const
{
    intel_reference_output_type outElemVal = static_cast<const intel_reference_output_type*>(outputBuffer)[i];
    const intel_reference_output_type* refOutput = refOutputAssign<intel_reference_output_type>(configIndex);
    if (refOutput[i] != outElemVal)
    {
        // TODO: how it should notified? return or throw
        throw std::runtime_error("Wrong output");
    }
}

void SetupConvolutionModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    unsigned int ref_output_size = refSize[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        (configIndex == configPwlEnabled) ? compareReferenceValues<int16_t>(i, configIndex) : compareReferenceValues<int32_t>(i, configIndex);
    }
}

void SetupConvolutionModel::samplePwl(intel_pwl_segment_t *segments, uint32_t numberOfSegments)
{
    int64_t xBase = INT32_MIN;
    auto xBaseInc = UINT32_MAX / numberOfSegments;
    uint16_t yBase = uint16_t(INT32_MAX);
    auto yBaseInc = UINT16_MAX / numberOfSegments;
    for (auto i = uint32_t{ 0 }; i < numberOfSegments; i++, xBase += xBaseInc, yBase += yBaseInc)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = yBase;
        segments[i].slope = 1;
    }
}
void SetupConvolutionModel::sampleConvolutionLayer()
{
    int buf_size_filters = ALIGN64(sizeof(filters));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = ALIGN64(sizeof(regularBiases));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int16_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    if (pwlEnabled)
    {
        int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));
        bytes_requested += buf_size_pwl;
    }
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, nnet.nLayers, 0, &bytes_granted);

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

    if (pwlEnabled)
    {
        void* pinned_pwl = pinned_mem_ptr;
        pinned_mem_ptr += nSegments * sizeof(intel_pwl_segment_t);

        pwl.nSegments = nSegments;
        pwl.pSegments = reinterpret_cast<intel_pwl_segment_t*>(pinned_pwl);
        samplePwl(pwl.pSegments, pwl.nSegments);
    }
    else
    {
        pwl.nSegments = 0;
        pwl.pSegments = NULL;
    }

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
    convolution_layer.poolType = INTEL_NO_POOLING;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = outVecSz;
    nnet.pLayers[0].nOutputRows = nnet.nGroup;
    if (pwlEnabled)
    {
        nnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
        nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t); // activated
    }
    else
    {
        nnet.pLayers[0].pOutputsIntermediate = nullptr;
        nnet.pLayers[0].nBytesPerOutput = sizeof(int32_t);
    }
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerIntermediateOutput = sizeof(int32_t);
    nnet.pLayers[0].nLayerKind = INTEL_CONVOLUTIONAL;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &convolution_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;
}
