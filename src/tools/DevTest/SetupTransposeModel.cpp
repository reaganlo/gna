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

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupTransposeModel::SetupTransposeModel(DeviceController & deviceCtrl, int configIndex)
    : deviceController{ deviceCtrl }
{
    nnet.nGroup = groupingNum[configIndex];
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleTransposeLayer(configIndex);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, InputComponent, 0, inputBuffer);
    deviceController.BufferAdd(configId, OutputComponent, 0, outputBuffer);
}

SetupTransposeModel::~SetupTransposeModel()
{
    deviceController.Free(memory);
    free(nnet.pLayers);

    deviceController.ModelRelease(modelId);
}

void SetupTransposeModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    int ref_output_size = refSize[configIndex];
    const int16_t * ref_output = refOutputAssign[configIndex];
    for (int i = 0; i < ref_output_size; ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupTransposeModel::sampleTransposeLayer(int configIndex)
{
    int buf_size_inputs = ALIGN64(inputsSize[configIndex]);
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum[configIndex] * sizeof(int32_t));

    uint32_t bytes_requested = buf_size_inputs + buf_size_outputs;
    uint32_t bytes_granted;

    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs[configIndex], inputsSize[configIndex]);
    pinned_mem_ptr += buf_size_inputs;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = nnet.nGroup;
    nnet.pLayers[0].nOutputRows = outVecSz;
    nnet.pLayers[0].nBytesPerInput = GNA_INT16;
    nnet.pLayers[0].nBytesPerOutput = GNA_INT16;
    nnet.pLayers[0].nBytesPerIntermediateOutput = GNA_INT32;
    nnet.pLayers[0].operation = INTEL_INTERLEAVE;
    nnet.pLayers[0].mode = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;
}
