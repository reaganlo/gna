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

#include "SetupGmmModel.h"

namespace
{
const int layersNum = 1;
const int groupingNum = 1;
const int inVecSz = 24;
const int outVecSz = 24;

int16_t weights[4 * 32] = {                                          // sample weight matrix (8 rows, 16 cols)
     1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
     1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
     1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
     1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
};

uint8_t inputs[2 * 32] = {
    1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
    1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,

};

int32_t biases[32] = {      // sample bias vector, will get added to each of the four output vectors
      1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
};

const uint32_t alIndices[outVecSz / 2]
{
    0, 2, 4, 7
};

const int32_t ref_output[outVecSz * groupingNum] =
{
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
}

SetupGmmModel::SetupGmmModel(DeviceController & deviceCtrl, bool activeListEn)
    : deviceController{deviceCtrl},
      activeListEnabled{activeListEn}
{
    sampleGmmLayer(nnet);

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, 0, outputBuffer);

    if (activeListEnabled)
    {
        deviceController.ActiveListAdd(configId, 0, indicesCount, indices);
    }
}

SetupGmmModel::~SetupGmmModel()
{
    deviceController.Free();

    free(nnet.pLayers->pLayerStruct);
    free(nnet.pLayers);
}

void SetupGmmModel::checkReferenceOutput(int modelIndex, int configIndex) const
{
    for (int i = 0; i < sizeof(ref_output) / sizeof(int32_t); ++i)
    {
        int32_t outElemVal = static_cast<const int32_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupGmmModel::sampleGmmLayer(intel_nnet_type_t& nnet)
{
    nnet.nGroup = groupingNum;
    nnet.nLayers = layersNum;
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));

    uint32_t stateCount = 8;

    int buf_size_weights     = ALIGN64(sizeof(weights)); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs      = ALIGN64(sizeof(inputs));
    int buf_size_biases      = ALIGN64(sizeof(biases));
    int buf_size_outputs     = ALIGN64(2*2*32 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)
    int buf_size_tmp_outputs = ALIGN64(2*2*32 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)

    // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    if(activeListEnabled) 
    {
        indicesCount = stateCount / 2;
        uint32_t buf_size_indices = indicesCount * sizeof(uint32_t);
        bytes_requested += buf_size_indices;
    }
    uint32_t bytes_granted;

    // call GNAAlloc (obtains pinned memory shared with the device)
    uint8_t* pinned_mem_ptr = deviceController.Alloc(bytes_requested, nnet.nLayers, 1, &bytes_granted);

    int16_t *pinned_weights = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_weights, weights, sizeof(weights));   // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    inputBuffer = (int16_t*)pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));      // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;                  // fast-forwards current pinned memory pointer to the next free block

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, biases, sizeof(biases));      // puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases;                  // fast-forwards current pinned memory pointer to the next free block

    outputBuffer = ((int16_t*)pinned_mem_ptr) + 32;
    pinned_mem_ptr += buf_size_outputs;                 // fast-forwards the current pinned memory pointer by the space needed for outputs

    int32_t *pinned_tmp_outputs = (int32_t*)pinned_mem_ptr;      // the last free block will be used for GNA's scratch pad
    pinned_mem_ptr += buf_size_tmp_outputs;

    if (activeListEnabled)
    {
        size_t indicesSize = indicesCount * sizeof(uint32_t);
        indices = (uint32_t*)pinned_mem_ptr;
        memcpy(indices, alIndices, indicesSize);
        pinned_mem_ptr += indicesSize;
    }

    gna_gmm_layer *gmm = (gna_gmm_layer*)calloc(1, sizeof(gna_gmm_layer));
    gmm->data.gaussianConstants = (uint32_t*)pinned_biases;
    gmm->data.inverseCovariancesForMaxMix16 = (uint16_t*)pinned_weights;
    gmm->data.meanValues = (uint8_t*)pinned_weights;
    gmm->config.layout = GMM_LAYOUT_FLAT;
    gmm->config.maximumScore = UINT32_MAX;
    gmm->config.mixtureComponentCount = 1;
    //gmm->config.mode = GNA_MAXMIX16;
    gmm->config.mode = GNA_MAXMIX8;
    gmm->config.stateCount = stateCount;

    nnet.pLayers[0].nInputColumns = inVecSz;
    nnet.pLayers[0].nInputRows = nnet.nGroup;
    nnet.pLayers[0].nOutputColumns = outVecSz;
    nnet.pLayers[0].nOutputRows = nnet.nGroup;
    nnet.pLayers[0].nBytesPerInput = 1;
    nnet.pLayers[0].nBytesPerOutput = 4;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet.pLayers[0].nBytesPerIntermediateOutput = 4; // this is always 4 bytes
    nnet.pLayers[0].type = INTEL_INPUT_OUTPUT;
    nnet.pLayers[0].nLayerKind = INTEL_GMM;
    nnet.pLayers[0].pLayerStruct = gmm;
    nnet.pLayers[0].pInputs = nullptr;
    nnet.pLayers[0].pOutputsIntermediate = nullptr;
    nnet.pLayers[0].pOutputs = nullptr;
}
