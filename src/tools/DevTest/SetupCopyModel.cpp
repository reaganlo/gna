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

#include "SetupCopyModel.h"

SetupCopyModel::SetupCopyModel(DeviceController & deviceCtrl, uint32_t nCopyColumns, uint32_t nCopyRows)
    : deviceController{deviceCtrl}
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    sampleCopyLayer(nCopyColumns, nCopyRows);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);
}

SetupCopyModel::~SetupCopyModel()
{
    deviceController.Free();

    free(nnet.pLayers);
}

void SetupCopyModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    unsigned int ref_output_size = refSize[configIndex];
    const int16_t * ref_output = refOutputAssign[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupCopyModel::sampleCopyLayer(uint32_t nCopyColumns, uint32_t nCopyRows)
{
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));

    uint32_t bytes_requested = buf_size_inputs + buf_size_outputs;
    uint32_t bytes_granted;

    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, nnet.nLayers, 0, &bytes_granted);

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    copy_layer.nCopyCols = nCopyColumns;
    copy_layer.nCopyRows = nCopyRows;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = outVecSz;
    nnet.pLayers[0].nOutputRows = nnet.nGroup;
    nnet.pLayers[0].nBytesPerInput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerOutput = sizeof(int16_t);
    nnet.pLayers[0].nBytesPerIntermediateOutput = 4;
    nnet.pLayers[0].nLayerKind = INTEL_COPY;
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].pLayerStruct = &copy_layer;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;
}
