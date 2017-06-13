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

#include "SetupTransposeModel.h"

namespace
{
const int layersNum = 1;
const int groupingNum = 4;
const int inVecSz = 16;
const int outVecSz = 16;

const int16_t inputs[groupingNum * inVecSz] = {
    -5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
    2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
    -9, -5, -8,  5, 0, -1,  3,  9, 0,  8,  1, -2, -9,  8,  0, -7,
    -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2
};

const int16_t ref_output[outVecSz * groupingNum] =
{
    -5, 2, -9, -9,
    9, -4, -5, -8, 
    -7, 9, -8, -1,
    4, 8, 5, -4,
    5, -5, 0, -3,
    -4, -1, -1, -7,
    -7, 2, 3, -2,
    4, 9, 9, 3,
    0, -8, 0, -8,
    7, -8, 8, 0,
    1, 8, 1, 1,
    -7, 1, -2, 3,
    1, -7, -9, -4,
    6, 2, 8, -6,
    7, -1, 0, -8,
    9, -1, -7, -2
};
}

SetupTransposeModel::SetupTransposeModel(DeviceController & deviceCtrl)
    : deviceController{deviceCtrl}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleTransposeLayer();

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupTransposeModel::~SetupTransposeModel()
{
    deviceController.ModelRelease(modelId);
    deviceController.Free();

    free(nnet.pLayers);
}

void SetupTransposeModel::checkReferenceOutput() const
{
    for (int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::exception("Wrong output");
        }
    }
}

void SetupTransposeModel::sampleTransposeLayer()
{
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));

    uint32_t bytes_requested = buf_size_inputs + buf_size_outputs;
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, &bytes_granted);

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = nnet.nGroup;
    nnet.pLayers[0].nOutputRows = outVecSz;
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerIntermediateOutput = 4;
    nnet.pLayers[0].nLayerKind = INTEL_INTERLEAVE;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;
}
