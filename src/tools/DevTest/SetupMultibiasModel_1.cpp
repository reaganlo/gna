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

#include "SetupMultibiasModel_1.h"

namespace
{
const int layersNum = 1;
const int groupingNum = 4;
const int inVecSz = 16;
const int outVecSz = 8;

const int8_t weights_1B[outVecSz * inVecSz] =
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

const int16_t weights_2B[outVecSz * inVecSz] =
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

const int16_t inputs[inVecSz * groupingNum] = {
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

const intel_bias_t regularBiases[outVecSz*groupingNum] = {
    5,5,5,5,
    4,4,4,4,
    -2,-2,-2,-2,
    5,5,5,5,
    -7,-7,-7,-7,
    -5,-5,-5,-5,
    4,4,4,4,
    -1,-1,-1,-1,
};

const  intel_compound_bias_t compoundBiases[outVecSz*groupingNum] =
{
    {5,1,{0}},{5,1,{0}},{5,1,{0}},{5,1,{0}},
    {4,1,{0}},{4,1,{0}},{4,1,{0}},{4,1,{0}},
    {-2,1,{0}},{-2,1,{0}},{-2,1,{0}},{-2,1,{0}},
    {5,1,{0}},{5,1,{0}},{5,1,{0}},{5,1,{0}},
    {-7,1,{0}},{-7,1,{0}},{-7,1,{0}},{-7,1,{0}},
    {-5,1,{0}},{-5,1,{0}},{-5,1,{0}},{-5,1,{0}},
    {4,1,{0}},{4,1,{0}},{4,1,{0}},{4,1,{0}},
    {-1,1,{0}},{-1,1,{0}},{-1,1,{0}},{-1,1,{0}},
};

const int32_t ref_output[outVecSz * groupingNum] =
{
    -177, -85, 29, 28,
    96, -173, 25, 252,
    -160, 274, 157, -29,
    48, -60, 158, -29,
    26, -2, -44, -251,
    -173, -70, -1, -323,
    99, 144, 38, -63,
    20, 56, -103, 10
};

const uint32_t alIndices[outVecSz / 2]
{
    0, 2, 4, 7
};
}

SetupMultibiasModel_1::SetupMultibiasModel_1(DeviceController & deviceCtrl, bool wght2B, bool pwlEn)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{wght2B},
    pwlEnabled{pwlEn}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleAffineLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupMultibiasModel_1::~SetupMultibiasModel_1()
{
    deviceController.ModelRelease(modelId);
    deviceController.Free();

    free(nnet.pLayers);
}

void SetupMultibiasModel_1::checkReferenceOutput() const
{
    for (int i = 0; i < sizeof(ref_output) / sizeof(int32_t); ++i)
    {
        int32_t outElemVal = static_cast<const int32_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::exception("Wrong output");
        }
    }
}

void SetupMultibiasModel_1::sampleAffineLayer()
{
    int buf_size_weights = weightsAre2Bytes ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B));
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = weightsAre2Bytes ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases));
    int buf_size_weight_scales = weightsAre2Bytes ? 0 : ALIGN64(sizeof(compoundBiases));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_tmp_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    int buf_size_pwl = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_weight_scales + buf_size_outputs + buf_size_tmp_outputs;

    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, &bytes_granted);

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

    intel_compound_bias_t *pinned_weight_scales = nullptr;
    if (!weightsAre2Bytes)
    {
        pinned_weight_scales = (intel_compound_bias_t*)pinned_mem_ptr;
        pinned_mem_ptr += buf_size_weight_scales;
    }

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    void *tmp_outputs;
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
        pwl.pSegments = NULL;
    }

    multibias_func.nBytesPerWeight = weightsAre2Bytes ? 2 : 1;
    multibias_func.pWeights = pinned_weights;
    multibias_func.pBiases = pinned_biases;
    multibias_func.biasVectorCount = 4;
    multibias_func.biasVectorIndex = 1;
    multibias_func.weightScaleFactors = pinned_weight_scales;

    multibias_layer.affine = multibias_func;
    multibias_layer.pwl = pwl;

    nnet.pLayers[0].nInputColumns = nnet.nGroup;
    nnet.pLayers[0].nInputRows = inVecSz;
    nnet.pLayers[0].nOutputColumns = nnet.nGroup;
    nnet.pLayers[0].nOutputRows = outVecSz;
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerIntermediateOutput = 4;
    nnet.pLayers[0].nLayerKind = INTEL_AFFINE_MULTIBIAS;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &multibias_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;

    if (pwlEnabled)
    {
        nnet.pLayers[0].pOutputsIntermediate = tmp_outputs;
        nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t);
    }
    else
    {
        nnet.pLayers[0].pOutputsIntermediate = nullptr;
        nnet.pLayers[0].nBytesPerOutput = sizeof(int32_t);
    }
}

void SetupMultibiasModel_1::samplePwl(intel_pwl_segment_t *segments, uint32_t nSegments)
{
    auto xBase = INT32_MIN;
    auto xBaseInc = UINT32_MAX / nSegments;
    auto yBase = INT32_MAX;
    auto yBaseInc = UINT16_MAX / nSegments;
    for (auto i = 0ui32; i < nSegments; i++, xBase += xBaseInc, yBase += yBaseInc)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = yBase;
        segments[i].slope = 1;
    }
}
